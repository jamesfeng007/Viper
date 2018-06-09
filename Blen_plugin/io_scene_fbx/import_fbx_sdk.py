import ctypes
from ctypes import c_char_p, byref, POINTER, c_double, c_int, c_char
import array
import bpy
from mathutils import Matrix, Euler, Vector
from . import fbx_utils, data_types
from fbx_ie_lib import FBXImport, GlobalSettings, ObjectTransformProp, LayerElementInfo
from collections import namedtuple
from .fbx_utils import (
    units_blender_to_fbx_factor,
    FBX_FRAMERATES,
    units_convertor_iter,
)

if "bpy" in locals():
    import importlib
    if "fbx_utils" in locals():
        importlib.reload(fbx_utils)
    if "data_types" in locals():
        importlib.reload(data_types)        
        
FBXImportSettings = namedtuple("FBXImportSettings", (
    "global_matrix", "global_scale", "bake_space_transform", "use_custom_normals", "global_matrix_inv_transposed", "global_matrix_inv",
))

FBXTransformData = namedtuple("FBXTransformData", (
    "loc", "geom_loc",
    "rot", "rot_ofs", "rot_piv", "pre_rot", "pst_rot", "rot_ord", "rot_alt_mat", "geom_rot",
    "sca", "sca_ofs", "sca_piv", "geom_sca",
))

FBXElem = namedtuple("FBXElem", ("id", "props", "props_type", "elems"))

convert_deg_to_rad_iter = units_convertor_iter("degree", "radian")

def blen_read_object_transform_do(transform_data):
    # This is a nightmare. FBX SDK uses Maya way to compute the transformation matrix of a node - utterly simple:
    #
    #     WorldTransform = ParentWorldTransform * T * Roff * Rp * Rpre * R * Rpost * Rp-1 * Soff * Sp * S * Sp-1
    #
    # Where all those terms are 4 x 4 matrices that contain:
    #     WorldTransform: Transformation matrix of the node in global space.
    #     ParentWorldTransform: Transformation matrix of the parent node in global space.
    #     T: Translation
    #     Roff: Rotation offset
    #     Rp: Rotation pivot
    #     Rpre: Pre-rotation
    #     R: Rotation
    #     Rpost: Post-rotation
    #     Rp-1: Inverse of the rotation pivot
    #     Soff: Scaling offset
    #     Sp: Scaling pivot
    #     S: Scaling
    #     Sp-1: Inverse of the scaling pivot
    #
    # But it was still too simple, and FBX notion of compatibility is... quite specific. So we also have to
    # support 3DSMax way:
    #
    #     WorldTransform = ParentWorldTransform * T * R * S * OT * OR * OS
    #
    # Where all those terms are 4 x 4 matrices that contain:
    #     WorldTransform: Transformation matrix of the node in global space
    #     ParentWorldTransform: Transformation matrix of the parent node in global space
    #     T: Translation
    #     R: Rotation
    #     S: Scaling
    #     OT: Geometric transform translation
    #     OR: Geometric transform rotation
    #     OS: Geometric transform translation
    #
    # Notes:
    #     Geometric transformations ***are not inherited***: ParentWorldTransform does not contain the OT, OR, OS
    #     of WorldTransform's parent node.
    #
    # Taken from http://download.autodesk.com/us/fbx/20112/FBX_SDK_HELP/
    #            index.html?url=WS1a9193826455f5ff1f92379812724681e696651.htm,topicNumber=d0e7429

    # translation
    lcl_translation = Matrix.Translation(transform_data.loc)
    geom_loc = Matrix.Translation(transform_data.geom_loc)

    # rotation
    to_rot = lambda rot, rot_ord: Euler(convert_deg_to_rad_iter(rot), rot_ord).to_matrix().to_4x4()
    lcl_rot = to_rot(transform_data.rot, transform_data.rot_ord) * transform_data.rot_alt_mat
    pre_rot = to_rot(transform_data.pre_rot, transform_data.rot_ord)
    pst_rot = to_rot(transform_data.pst_rot, transform_data.rot_ord)
    geom_rot = to_rot(transform_data.geom_rot, transform_data.rot_ord)

    rot_ofs = Matrix.Translation(transform_data.rot_ofs)
    rot_piv = Matrix.Translation(transform_data.rot_piv)
    sca_ofs = Matrix.Translation(transform_data.sca_ofs)
    sca_piv = Matrix.Translation(transform_data.sca_piv)

    # scale
    lcl_scale = Matrix()
    lcl_scale[0][0], lcl_scale[1][1], lcl_scale[2][2] = transform_data.sca
    geom_scale = Matrix();
    geom_scale[0][0], geom_scale[1][1], geom_scale[2][2] = transform_data.geom_sca

    base_mat = (
        lcl_translation *
        rot_ofs *
        rot_piv *
        pre_rot *
        lcl_rot *
        pst_rot *
        rot_piv.inverted_safe() *
        sca_ofs *
        sca_piv *
        lcl_scale *
        sca_piv.inverted_safe()
    )
    geom_mat = geom_loc * geom_rot * geom_scale
    # We return mat without 'geometric transforms' too, because it is to be used for children, sigh...
    return (base_mat * geom_mat, base_mat, geom_mat)

# ### Import Utility class
class FbxImportHelperNode:
    __slots__ = (
        '_parent', 'anim_compensation_matrix', 'is_global_animation', 'armature_setup', 'armature', 'bind_matrix',
        'bl_bone', 'bl_data', 'bl_obj', 'bone_child_matrix', 'children', 'clusters',
        'fbx_elem', 'fbx_name', 'fbx_transform_data', 'fbx_type',
        'is_armature', 'has_bone_children', 'is_bone', 'is_root', 'is_leaf',
        'matrix', 'matrix_as_parent', 'matrix_geom', 'meshes', 'post_matrix', 'pre_matrix')
    
    def __init__(self, fbx_elem, bl_data, fbx_transform_data, is_bone):
        self.fbx_name = fbx_elem.props if fbx_elem else 'Unknown'
        self.fbx_type = fbx_elem.props_type if fbx_elem else None
        self.fbx_elem = fbx_elem
        self.bl_obj = None
        self.bl_data = bl_data
        self.bl_bone = None                     # Name of bone if this is a bone (this may be different to fbx_name if there was a name conflict in Blender!)
        self.fbx_transform_data = fbx_transform_data
        self.is_root = False
        self.is_bone = is_bone
        self.is_armature = False
        self.armature = None                    # For bones only, relevant armature node.
        self.has_bone_children = False          # True if the hierarchy below this node contains bones, important to support mixed hierarchies.
        self.is_leaf = False                    # True for leaf-bones added to the end of some bone chains to set the lengths.
        self.pre_matrix = None                  # correction matrix that needs to be applied before the FBX transform
        self.bind_matrix = None                 # for bones this is the matrix used to bind to the skin
        if fbx_transform_data:
            self.matrix, self.matrix_as_parent, self.matrix_geom = blen_read_object_transform_do(fbx_transform_data)
        else:
            self.matrix, self.matrix_as_parent, self.matrix_geom = (None, None, None)
        self.post_matrix = None                 # correction matrix that needs to be applied after the FBX transform
        self.bone_child_matrix = None           # Objects attached to a bone end not the beginning, this matrix corrects for that

        # XXX Those two are to handle the fact that rigged meshes are not linked to their armature in FBX, which implies
        #     that their animation is in global space (afaik...).
        #     This is actually not really solvable currently, since anim_compensation_matrix is not valid if armature
        #     itself is animated (we'd have to recompute global-to-local anim_compensation_matrix for each frame,
        #     and for each armature action... beyond being an insane work).
        #     Solution for now: do not read rigged meshes animations at all! sic...
        self.anim_compensation_matrix = None    # a mesh moved in the hierarchy may have a different local matrix. This compensates animations for this.
        self.is_global_animation = False

        self.meshes = None                      # List of meshes influenced by this bone.
        self.clusters = []                      # Deformer Cluster nodes
        self.armature_setup = {}                # mesh and armature matrix when the mesh was bound

        self._parent = None
        self.children = []
        
    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        if self._parent is not None:
            self._parent.children.remove(self)
        self._parent = value
        if self._parent is not None:
            self._parent.children.append(self)
            
    @property
    def ignore(self):
        # Separating leaf status from ignore status itself.
        # Currently they are equivalent, but this may change in future.
        return self.is_leaf        
            
    def print_info(self, indent=0):
        print(" " * indent + (self.fbx_name if self.fbx_name else "(Null)")
              + ("[root]" if self.is_root else "")
              + ("[leaf]" if self.is_leaf else "")
              + ("[ignore]" if self.ignore else "")
              + ("[armature]" if self.is_armature else "")
              + ("[bone]" if self.is_bone else "")
              + ("[HBC]" if self.has_bone_children else "")
              )
        for c in self.children:
            c.print_info(indent + 1)
            
    def get_matrix(self):
        matrix = self.matrix if self.matrix else Matrix()
        if self.pre_matrix:
            matrix = self.pre_matrix * matrix
        if self.post_matrix:
            matrix = matrix * self.post_matrix
        return matrix            
        
    def build_node_obj(self, settings):
        if self.bl_obj:
            return self.bl_obj

        if self.is_bone or not self.fbx_elem:
            return None

        # create when linking since we need object data
        elem_name_utf8 = self.fbx_name

        # Object data must be created already
        self.bl_obj = obj = bpy.data.objects.new(name=elem_name_utf8, object_data=self.bl_data)

        # ----
        # Misc Attributes

        obj.color[0:3] = (0.8, 0.8, 0.8)
        obj.hide = not bool(1.0)

        obj.matrix_basis = self.get_matrix()

        #if settings.use_custom_props:
            #blen_read_custom_properties(self.fbx_elem, obj, settings)

        return obj        
        
    def build_hierarchy(self, settings, scene):
        if self.is_armature:
            return None
        elif self.fbx_elem and not self.is_bone:
            obj = self.build_node_obj(settings)

            # walk through children
            for child in self.children:
                child.build_hierarchy(settings, scene)

            # instance in scene
            obj_base = scene.objects.link(obj)
            obj_base.select = True

            return obj
        else:
            for child in self.children:
                child.build_hierarchy(settings, scene)

            return None
        
    def link_hierarchy(self, settings, scene):
        if self.is_armature:
            return None
        elif self.bl_obj:
            obj = self.bl_obj

            # walk through children
            for child in self.children:
                child_obj = child.link_hierarchy(settings, scene)
                if child_obj:
                    child_obj.parent = obj

            return obj
        else:
            for child in self.children:
                child.link_hierarchy(settings, scene)

            return None
        
    def do_bake_transform(self, settings):
        return (settings.bake_space_transform and self.fbx_type in (b'Mesh', b'Null') and
                not self.is_armature and not self.is_bone)        
        
    def find_correction_matrix(self, settings, parent_correction_inv=None):
        if self.parent and (self.parent.is_root or self.parent.do_bake_transform(settings)):
            self.pre_matrix = settings.global_matrix

        if parent_correction_inv:
            self.pre_matrix = parent_correction_inv * (self.pre_matrix if self.pre_matrix else Matrix())

        correction_matrix = None

        self.post_matrix = correction_matrix

        if self.do_bake_transform(settings):
            self.post_matrix = settings.global_matrix_inv * (self.post_matrix if self.post_matrix else Matrix())

        # process children
        correction_matrix_inv = correction_matrix.inverted_safe() if correction_matrix else None
        for child in self.children:
            child.find_correction_matrix(settings, correction_matrix_inv)                
        
def blen_read_object_transform_preprocess(fbxSDKImport, index, rot_alt_mat, use_prepost_rot):
    const_vector_zero_3d = 0.0, 0.0, 0.0
    const_vector_one_3d = 1.0, 1.0, 1.0
        
    fbx_object_trans_prop = ObjectTransformProp()
    fbxSDKImport.get_mesh_object_transform_prop(index, byref(fbx_object_trans_prop))
    
    loc = [fbx_object_trans_prop.lclTranslation.x, fbx_object_trans_prop.lclTranslation.y, fbx_object_trans_prop.lclTranslation.z]
    rot = [fbx_object_trans_prop.lclRotation.x, fbx_object_trans_prop.lclRotation.y, fbx_object_trans_prop.lclRotation.z]
    sca = [fbx_object_trans_prop.lclScaling.x, fbx_object_trans_prop.lclScaling.y, fbx_object_trans_prop.lclScaling.z]
    
    geom_loc = [fbx_object_trans_prop.GeometricTranslation.x, fbx_object_trans_prop.GeometricTranslation.y, fbx_object_trans_prop.GeometricTranslation.z]
    geom_rot = [fbx_object_trans_prop.GeometricRotation.x, fbx_object_trans_prop.GeometricRotation.y, fbx_object_trans_prop.GeometricRotation.z]
    geom_sca = [fbx_object_trans_prop.GeometricScaling.x, fbx_object_trans_prop.GeometricScaling.y, fbx_object_trans_prop.GeometricScaling.z]
    
    rot_ofs = (fbx_object_trans_prop.RotationOffset.x, fbx_object_trans_prop.RotationOffset.y, fbx_object_trans_prop.RotationOffset.z)
    rot_piv = (fbx_object_trans_prop.RotationPivot.x, fbx_object_trans_prop.RotationPivot.y, fbx_object_trans_prop.RotationPivot.z)
    sca_ofs = (fbx_object_trans_prop.ScalingOffset.x, fbx_object_trans_prop.ScalingOffset.y, fbx_object_trans_prop.ScalingOffset.z)
    sca_piv = (fbx_object_trans_prop.ScalingPivot.x, fbx_object_trans_prop.ScalingPivot.y, fbx_object_trans_prop.ScalingPivot.z)
    
    is_rot_act = fbx_object_trans_prop.RotationActive
    
    if is_rot_act:
        if use_prepost_rot:
            pre_rot = (fbx_object_trans_prop.PreRotation.x, fbx_object_trans_prop.PreRotation.y, fbx_object_trans_prop.PreRotation.z)
            pst_rot = (fbx_object_trans_prop.PostRotation.x, fbx_object_trans_prop.PostRotation.y, fbx_object_trans_prop.PostRotation.z)
        else:
            pre_rot = const_vector_zero_3d
            pst_rot = const_vector_zero_3d
        rot_ord = {
            0: 'XYZ',
            1: 'XZY',
            2: 'YZX',
            3: 'YXZ',
            4: 'ZXY',
            5: 'ZYX',
            6: 'XYZ',  # XXX eSphericXYZ, not really supported...
            }.get(fbx_object_trans_prop.RotationOrder)
    else:
        pre_rot = const_vector_zero_3d
        pst_rot = const_vector_zero_3d
        rot_ord = 'XYZ'    
    
    return FBXTransformData(loc, geom_loc,
                            rot, rot_ofs, rot_piv, pre_rot, pst_rot, rot_ord, rot_alt_mat, geom_rot,
                            sca, sca_ofs, sca_piv, geom_sca)
    
def blen_read_geom_array_gen_direct(fbx_data, stride):
    fbx_data_len = len(fbx_data)
    return zip(*(range(fbx_data_len // stride), range(0, fbx_data_len, stride)))    
    
def blen_read_geom_array_setattr(generator, blen_data, blen_attr, fbx_data, stride, item_size, descr, xform):
    """Generic fbx_layer to blen_data setter, generator is expected to yield tuples (ble_idx, fbx_idx)."""
    max_idx = len(blen_data) - 1
    print_error = True

    def check_skip(blen_idx, fbx_idx):
        nonlocal print_error
        if fbx_idx < 0:  # Negative values mean 'skip'.
            return True
        if blen_idx > max_idx:
            if print_error:
                print("ERROR: too much data in this layer, compared to elements in mesh, skipping!")
                print_error = False
            return True
        return False

    if xform is not None:
        if isinstance(blen_data, list):
            if item_size == 1:
                def _process(blend_data, blen_attr, fbx_data, xform, item_size, blen_idx, fbx_idx):
                    blen_data[blen_idx] = xform(fbx_data[fbx_idx])
            else:
                def _process(blend_data, blen_attr, fbx_data, xform, item_size, blen_idx, fbx_idx):
                    blen_data[blen_idx] = xform(fbx_data[fbx_idx:fbx_idx + item_size])
        else:
            if item_size == 1:
                def _process(blend_data, blen_attr, fbx_data, xform, item_size, blen_idx, fbx_idx):
                    setattr(blen_data[blen_idx], blen_attr, xform(fbx_data[fbx_idx]))
            else:
                def _process(blend_data, blen_attr, fbx_data, xform, item_size, blen_idx, fbx_idx):
                    setattr(blen_data[blen_idx], blen_attr, xform(fbx_data[fbx_idx:fbx_idx + item_size]))
    else:
        if isinstance(blen_data, list):
            if item_size == 1:
                def _process(blend_data, blen_attr, fbx_data, xform, item_size, blen_idx, fbx_idx):
                    blen_data[blen_idx] = fbx_data[fbx_idx]
            else:
                def _process(blend_data, blen_attr, fbx_data, xform, item_size, blen_idx, fbx_idx):
                    blen_data[blen_idx] = fbx_data[fbx_idx:fbx_idx + item_size]
        else:
            if item_size == 1:
                def _process(blend_data, blen_attr, fbx_data, xform, item_size, blen_idx, fbx_idx):
                    setattr(blen_data[blen_idx], blen_attr, fbx_data[fbx_idx])
            else:
                def _process(blend_data, blen_attr, fbx_data, xform, item_size, blen_idx, fbx_idx):
                    setattr(blen_data[blen_idx], blen_attr, fbx_data[fbx_idx:fbx_idx + item_size])

    for blen_idx, fbx_idx in generator:
        if check_skip(blen_idx, fbx_idx):
            continue
        _process(blen_data, blen_attr, fbx_data, xform, item_size, blen_idx, fbx_idx)
    
def blen_read_geom_array_mapped_vert(
        mesh, blen_data, blen_attr,
        fbx_layer_data, fbx_layer_index,
        fbx_layer_mapping, fbx_layer_ref,
        stride, item_size, descr,
        xform=None, quiet=False,
        ):
    if fbx_layer_mapping == b'ByVertice':
        if fbx_layer_ref == b'Direct':
            assert(fbx_layer_index is None)
            blen_read_geom_array_setattr(blen_read_geom_array_gen_direct(fbx_layer_data, stride),
                                         blen_data, blen_attr, fbx_layer_data, stride, item_size, descr, xform)
            return True
        blen_read_geom_array_error_ref(descr, fbx_layer_ref, quiet)
    elif fbx_layer_mapping == b'AllSame':
        if fbx_layer_ref == b'IndexToDirect':
            assert(fbx_layer_index is None)
            blen_read_geom_array_setattr(blen_read_geom_array_gen_allsame(len(blen_data)),
                                         blen_data, blen_attr, fbx_layer_data, stride, item_size, descr, xform)
            return True
        blen_read_geom_array_error_ref(descr, fbx_layer_ref, quiet)
    else:
        blen_read_geom_array_error_mapping(descr, fbx_layer_mapping, quiet)

    return False
    
def blen_read_geom_array_mapped_polygon(
        mesh, blen_data, blen_attr,
        fbx_layer_data, fbx_layer_index,
        fbx_layer_mapping, fbx_layer_ref,
        stride, item_size, descr,
        xform=None, quiet=False,
        ):
    if fbx_layer_mapping == b'ByPolygon':
        if fbx_layer_ref == b'IndexToDirect':
            # XXX Looks like we often get no fbx_layer_index in this case, shall not happen but happens...
            #     We fallback to 'Direct' mapping in this case.
            #~ assert(fbx_layer_index is not None)
            if fbx_layer_index is None:
                blen_read_geom_array_setattr(blen_read_geom_array_gen_direct(fbx_layer_data, stride),
                                             blen_data, blen_attr, fbx_layer_data, stride, item_size, descr, xform)
            else:
                blen_read_geom_array_setattr(blen_read_geom_array_gen_indextodirect(fbx_layer_index, stride),
                                             blen_data, blen_attr, fbx_layer_data, stride, item_size, descr, xform)
            return True
        elif fbx_layer_ref == b'Direct':
            blen_read_geom_array_setattr(blen_read_geom_array_gen_direct(fbx_layer_data, stride),
                                         blen_data, blen_attr, fbx_layer_data, stride, item_size, descr, xform)
            return True
        blen_read_geom_array_error_ref(descr, fbx_layer_ref, quiet)
    elif fbx_layer_mapping == b'AllSame':
        if fbx_layer_ref == b'IndexToDirect':
            assert(fbx_layer_index is None)
            blen_read_geom_array_setattr(blen_read_geom_array_gen_allsame(len(blen_data)),
                                         blen_data, blen_attr, fbx_layer_data, stride, item_size, descr, xform)
            return True
        blen_read_geom_array_error_ref(descr, fbx_layer_ref, quiet)
    else:
        blen_read_geom_array_error_mapping(descr, fbx_layer_mapping, quiet)

    return False
    
def blen_read_geom_array_mapped_polyloop(
        mesh, blen_data, blen_attr,
        fbx_layer_data, fbx_layer_index,
        fbx_layer_mapping, fbx_layer_ref,
        stride, item_size, descr,
        xform=None, quiet=False,
        ):
    if fbx_layer_mapping == b'ByPolygonVertex':
        if fbx_layer_ref == b'IndexToDirect':
            # XXX Looks like we often get no fbx_layer_index in this case, shall not happen but happens...
            #     We fallback to 'Direct' mapping in this case.
            #~ assert(fbx_layer_index is not None)
            if fbx_layer_index is None:
                blen_read_geom_array_setattr(blen_read_geom_array_gen_direct(fbx_layer_data, stride),
                                             blen_data, blen_attr, fbx_layer_data, stride, item_size, descr, xform)
            else:
                blen_read_geom_array_setattr(blen_read_geom_array_gen_indextodirect(fbx_layer_index, stride),
                                             blen_data, blen_attr, fbx_layer_data, stride, item_size, descr, xform)
            return True
        elif fbx_layer_ref == b'Direct':
            blen_read_geom_array_setattr(blen_read_geom_array_gen_direct(fbx_layer_data, stride),
                                         blen_data, blen_attr, fbx_layer_data, stride, item_size, descr, xform)
            return True
        blen_read_geom_array_error_ref(descr, fbx_layer_ref, quiet)
    elif fbx_layer_mapping == b'ByVertice':
        if fbx_layer_ref == b'Direct':
            assert(fbx_layer_index is None)
            blen_read_geom_array_setattr(blen_read_geom_array_gen_direct_looptovert(mesh, fbx_layer_data, stride),
                                         blen_data, blen_attr, fbx_layer_data, stride, item_size, descr, xform)
            return True
        blen_read_geom_array_error_ref(descr, fbx_layer_ref, quiet)
    elif fbx_layer_mapping == b'AllSame':
        if fbx_layer_ref == b'IndexToDirect':
            assert(fbx_layer_index is None)
            blen_read_geom_array_setattr(blen_read_geom_array_gen_allsame(len(blen_data)),
                                         blen_data, blen_attr, fbx_layer_data, stride, item_size, descr, xform)
            return True
        blen_read_geom_array_error_ref(descr, fbx_layer_ref, quiet)
    else:
        blen_read_geom_array_error_mapping(descr, fbx_layer_mapping, quiet)

    return False    
    
def blen_read_geom_layer_normal(fbxSDKImport, mesh_index, mesh, xform=None):
    normal_size = fbxSDKImport.get_mesh_normal_size(mesh_index)
    normals = (c_double * (normal_size * 3))()
    fbx_layerInfo = LayerElementInfo()
    fbxSDKImport.get_mesh_normal(mesh_index, byref(normals), len(normals), byref(fbx_layerInfo))
    fbx_nors = array.array(data_types.ARRAY_FLOAT64, ())
    for n in normals:
        fbx_nors.append(n)
        
    layer_id = b'Normals'
    fbx_layer_data = fbx_nors
    fbx_layer_index = None
    fbx_layer_mapping = fbx_layerInfo.MappingType
    fbx_layer_ref = fbx_layerInfo.RefType
    
    # try loops, then vertices.
    tries = ((mesh.loops, "Loops", False, blen_read_geom_array_mapped_polyloop),
             (mesh.polygons, "Polygons", True, blen_read_geom_array_mapped_polygon),
             (mesh.vertices, "Vertices", True, blen_read_geom_array_mapped_vert))
    for blen_data, blen_data_type, is_fake, func in tries:
        bdata = [None] * len(blen_data) if is_fake else blen_data
        if func(mesh, bdata, "normal",
                fbx_layer_data, fbx_layer_index, fbx_layer_mapping, fbx_layer_ref, 3, 3, layer_id, xform, True):
            if blen_data_type is "Polygons":
                for pidx, p in enumerate(mesh.polygons):
                    for lidx in range(p.loop_start, p.loop_start + p.loop_total):
                        mesh.loops[lidx].normal[:] = bdata[pidx]
            elif blen_data_type is "Vertices":
                # We have to copy vnors to lnors! Far from elegant, but simple.
                for l in mesh.loops:
                    l.normal[:] = bdata[l.vertex_index]
            return True    
    return False

def blen_read_geom(fbxSDKImport, settings, mesh_index):
    from itertools import chain    
    geom_mat_co = settings.global_matrix if settings.bake_space_transform else None
    # We need to apply the inverse transpose of the global matrix when transforming normals.
    geom_mat_no = Matrix(settings.global_matrix_inv_transposed) if settings.bake_space_transform else None
    if geom_mat_no is not None:
        # Remove translation & scaling!
        geom_mat_no.translation = Vector()
        geom_mat_no.normalize()    
    
    mesh_name = fbxSDKImport.get_mesh_name(mesh_index)
    vertice_size = fbxSDKImport.get_mesh_vertice_size(mesh_index)
    vertices = (c_double * (vertice_size * 3))()
    fbxSDKImport.get_mesh_vertice(mesh_index, byref(vertices), len(vertices))
    fbx_verts = array.array(data_types.ARRAY_FLOAT64, ())
    print("mesh_name: %s" % mesh_name)
    for v in vertices:
        fbx_verts.append(v)
     
    if geom_mat_co is not None:
        def _vcos_transformed_gen(raw_cos, m=None):
            # Note: we could most likely get much better performances with numpy, but will leave this as TODO for now.
            return chain(*(m * Vector(v) for v in zip(*(iter(raw_cos),) * 3)))
        fbx_verts = array.array(fbx_verts.typecode, _vcos_transformed_gen(fbx_verts, geom_mat_co))   
    
    indice_size = fbxSDKImport.get_mesh_indice_size(mesh_index)
    indices = (c_int * indice_size)()
    fbxSDKImport.get_mesh_indice(mesh_index, byref(indices), len(indices))
    fbx_polys = array.array('l', ())
    for indice in indices:
        fbx_polys.append(indice)
        
    elem_name_utf8 = mesh_name.decode('utf-8')
    mesh = bpy.data.meshes.new(name=elem_name_utf8)
    mesh.vertices.add(len(fbx_verts) // 3)
    mesh.vertices.foreach_set("co", fbx_verts)
    
    if fbx_polys:
        mesh.loops.add(len(fbx_polys))
        poly_loop_starts = []
        poly_loop_totals = []
        poly_loop_prev = 0
        for i, l in enumerate(mesh.loops):
            index = fbx_polys[i]
            if index < 0:
                poly_loop_starts.append(poly_loop_prev)
                poly_loop_totals.append((i - poly_loop_prev) + 1)
                poly_loop_prev = i + 1
                index ^= -1
            l.vertex_index = index

        mesh.polygons.add(len(poly_loop_starts))
        mesh.polygons.foreach_set("loop_start", poly_loop_starts)
        mesh.polygons.foreach_set("loop_total", poly_loop_totals)
        
    ok_normals = False
    if settings.use_custom_normals:
        mesh.create_normals_split()
        if geom_mat_no is None:
            ok_normals = blen_read_geom_layer_normal(fbxSDKImport, mesh_index, mesh)
        else:
            def nortrans(v):
                return geom_mat_no * Vector(v)
            ok_normals = blen_read_geom_layer_normal(fbxSDKImport, mesh_index, mesh, nortrans)

    mesh.validate(clean_customdata=False)  # *Very* important to not remove lnors here!
    
    print("ok_normals")
    print(ok_normals)
    if ok_normals:
        clnors = array.array('f', [0.0] * (len(mesh.loops) * 3))
        mesh.loops.foreach_get("normal", clnors)
        '''
        if not ok_smooth:
            mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
            ok_smooth = True
        '''

        mesh.normals_split_custom_set(tuple(zip(*(iter(clnors),) * 3)))
        mesh.use_auto_smooth = True
        mesh.show_edge_sharp = True
    else:
        mesh.calc_normals()

    if settings.use_custom_normals:
        mesh.free_normals_split()    
        
    fbx_obj = FBXElem(
        1, elem_name_utf8, b"Mesh", None
    )
        
    return mesh, fbx_obj

def load(operator, context, filepath="",
         use_manual_orientation=False,
         axis_forward='-Z',
         axis_up='Y',
         global_scale=1.0,
         bake_space_transform=False,
         use_custom_normals=True,
         use_cycles=True,
         use_image_search=False,
         use_alpha_decals=False,
         decal_offset=0.0,
         use_anim=True,
         anim_offset=1.0,
         use_custom_props=True,
         use_custom_props_enum_as_string=True,
         ignore_leaf_bones=False,
         force_connect_children=False,
         automatic_bone_orientation=False,
         primary_bone_axis='Y',
         secondary_bone_axis='X',
         use_prepost_rot=True):
    
    import os
    import time
    from bpy_extras.io_utils import axis_conversion    
    
    print("filepath: %s" % filepath)
    
    fbxSDKImport = FBXImport()
    fbxSDKImport.fbx_import(c_char_p(filepath.encode('utf-8')))
    
    #fbxSDKImport.print_mesh()
    
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        
    # deselect all
    if bpy.ops.object.select_all.poll():
        bpy.ops.object.select_all(action='DESELECT')
        
    scene = context.scene
        
    fbx_global_settings = GlobalSettings()
    fbxSDKImport.get_global_settings(byref(fbx_global_settings))
    unit_scale = fbx_global_settings.UnitScaleFactor
    unit_scale_org = fbx_global_settings.OriginalUnitScaleFactor
    global_scale *= (unit_scale / units_blender_to_fbx_factor(context.scene))
    
    axis_forward = fbx_global_settings.AxisForward.decode('utf-8')
    axis_up = fbx_global_settings.AxisUp.decode('utf-8')
    
    global_matrix = (Matrix.Scale(global_scale, 4) *
                 axis_conversion(from_forward=axis_forward, from_up=axis_up).to_4x4()) 
    # To cancel out unwanted rotation/scale on nodes.
    global_matrix_inv = global_matrix.inverted()
    # For transforming mesh normals.
    global_matrix_inv_transposed = global_matrix_inv.transposed()
    
    # Compute bone correction matrix
    bone_correction_matrix = None  # None means no correction/identity
    if not automatic_bone_orientation:
        if (primary_bone_axis, secondary_bone_axis) != ('Y', 'X'):
            bone_correction_matrix = axis_conversion(from_forward='X',
                                                     from_up='Y',
                                                     to_forward=secondary_bone_axis,
                                                     to_up=primary_bone_axis,
                                                     ).to_4x4()
    custom_fps = fbx_global_settings.CustomFrameRate
    time_mode = fbx_global_settings.TimeMode
    real_fps = {eid: val for val, eid in FBX_FRAMERATES[1:]}.get(time_mode, custom_fps)
    if real_fps <= 0.0:
        real_fps = 25.0
    scene.render.fps = round(real_fps)
    scene.render.fps_base = scene.render.fps / real_fps
    
    settings = FBXImportSettings(
        global_matrix, global_scale, bake_space_transform, use_custom_normals, global_matrix_inv_transposed, global_matrix_inv
    )
    
    fbx_helper_nodes = {}
    # create scene root
    fbx_helper_nodes[0] = root_helper = FbxImportHelperNode(None, None, None, False)
    root_helper.is_root = True    
    
    mesh_count = fbxSDKImport.get_mesh_count()
    for i in range(mesh_count):
        bl_data, fbx_obj = blen_read_geom(fbxSDKImport, settings, i)
        transform_data = blen_read_object_transform_preprocess(fbxSDKImport, i, Matrix(), use_prepost_rot)

        fbx_helper_nodes[1] = FbxImportHelperNode(fbx_obj, bl_data, transform_data, False)
        
    parent = fbx_helper_nodes.get(0)
    child = fbx_helper_nodes.get(1)
    child.parent = parent
            
    root_helper.find_correction_matrix(settings)
    root_helper.build_hierarchy(settings, scene)
    root_helper.link_hierarchy(settings, scene)
    root_helper.print_info(0)

    return {'FINISHED'}
    
    
        