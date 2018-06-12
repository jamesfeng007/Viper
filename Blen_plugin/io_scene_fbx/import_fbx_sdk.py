import ctypes
from ctypes import c_char_p, byref, POINTER, c_double, c_int, c_char
import array
import bpy
from mathutils import Matrix, Euler, Vector
from . import fbx_utils, data_types
from fbx_ie_lib import FBXImport, GlobalSettings, ObjectTransformProp, LayerElementInfo, Vector3, UInt64Vector2
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
    "cycles_material_wrap_map", "use_cycles", "image_cache", "use_image_search",
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
        
def blen_read_model_transform_preprocess(fbxSDKImport, model_index, rot_alt_mat, use_prepost_rot):
    const_vector_zero_3d = 0.0, 0.0, 0.0
    const_vector_one_3d = 1.0, 1.0, 1.0
        
    fbx_object_trans_prop = ObjectTransformProp()
    fbxSDKImport.get_model_transform_prop(model_index, byref(fbx_object_trans_prop))
    
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
        
    elem_name_utf8 = fbxSDKImport.get_model_name(model_index).decode('utf-8')
        
    fbx_obj = FBXElem(
        b'Model', elem_name_utf8, b"Model", None
    )        
    
    return fbx_obj, FBXTransformData(loc, geom_loc,
                            rot, rot_ofs, rot_piv, pre_rot, pst_rot, rot_ord, rot_alt_mat, geom_rot,
                            sca, sca_ofs, sca_piv, geom_sca)
    
def blen_read_geom_array_gen_allsame(data_len):
    return zip(*(range(data_len), (0,) * data_len))    
    
def blen_read_geom_array_gen_direct(fbx_data, stride):
    fbx_data_len = len(fbx_data)
    return zip(*(range(fbx_data_len // stride), range(0, fbx_data_len, stride)))

def blen_read_geom_array_gen_indextodirect(fbx_layer_index, stride):
    return ((bi, fi * stride) for bi, fi in enumerate(fbx_layer_index))
    
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

def blen_read_geom_array_mapped_edge(
        mesh, blen_data, blen_attr,
        fbx_layer_data, fbx_layer_index,
        fbx_layer_mapping, fbx_layer_ref,
        stride, item_size, descr,
        xform=None, quiet=False,
        ):
    if fbx_layer_mapping == b'ByEdge':
        if fbx_layer_ref == b'Direct':
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

def blen_read_geom_layer_uv(fbxSDKImport, mesh_index, mesh):
    uv_info_size = fbxSDKImport.get_mesh_uv_info_size(mesh_index)
    layer_id = b'LayerElementUV'
    for uv_index in range(uv_info_size):
        fbx_layerInfo = LayerElementInfo()
        uv_name = fbxSDKImport.get_uv_info_name(mesh_index, uv_index, byref(fbx_layerInfo))
        
        fbx_layer_name = uv_name.decode('utf-8')
        fbx_layer_mapping = fbx_layerInfo.MappingType
        fbx_layer_ref = fbx_layerInfo.RefType
        
        uv_indice_size = fbxSDKImport.get_mesh_uv_indice_size(mesh_index, uv_index)
        indices = (c_int * uv_indice_size)()
        fbxSDKImport.get_mesh_uv_indice(mesh_index, uv_index, byref(indices), len(indices))
        fbx_layer_index = array.array('l', ())
        for indice in indices:
            fbx_layer_index.append(indice)
            
        uv_vertice_size = fbxSDKImport.get_mesh_uv_vertice_size(mesh_index, uv_index)
        vertices = (c_double * (uv_vertice_size * 2))()
        fbxSDKImport.get_mesh_uv_vertice(mesh_index, uv_index, byref(vertices), len(vertices))
        fbx_layer_data = array.array(data_types.ARRAY_FLOAT64, ())
        for v in vertices:
            fbx_layer_data.append(v)
            
        uv_tex = mesh.uv_textures.new(name=fbx_layer_name)
        uv_lay = mesh.uv_layers[-1]
        blen_data = uv_lay.data

        # some valid files omit this data
        if len(fbx_layer_data) == 0:
            print("%r %r missing data" % (layer_id, fbx_layer_name))
            continue

        blen_read_geom_array_mapped_polyloop(
            mesh, blen_data, "uv",
            fbx_layer_data, fbx_layer_index,
            fbx_layer_mapping, fbx_layer_ref,
            2, 2, layer_id,
            )

def blen_read_geom_layer_smooth(fbxSDKImport, mesh_index, mesh):
    smooth_size = fbxSDKImport.get_mesh_smoothing_size(mesh_index)
    if smooth_size == 0:
        return False
        
    smoothings = (c_int * smooth_size)()
    fbx_layerInfo = LayerElementInfo()
    fbxSDKImport.get_mesh_smoothing(mesh_index, byref(smoothings), len(smoothings), byref(fbx_layerInfo))
    fbx_layer_data = array.array('l', ())
    for s in smoothings:
        fbx_layer_data.append(s)
        
    if len(fbx_layer_data) == 0:
        return False        
    
    layer_id = b'Smoothing'
    fbx_layer_mapping = fbx_layerInfo.MappingType
    fbx_layer_ref = fbx_layerInfo.RefType
    
    if fbx_layer_mapping == b'ByEdge':
        # some models have bad edge data, we cant use this info...
        if not mesh.edges:
            print("warning skipping sharp edges data, no valid edges...")
            return False

        blen_data = mesh.edges
        blen_read_geom_array_mapped_edge(
            mesh, blen_data, "use_edge_sharp",
            fbx_layer_data, None,
            fbx_layer_mapping, fbx_layer_ref,
            1, 1, layer_id,
            xform=lambda s: not s,
            )
        # We only set sharp edges here, not face smoothing itself...
        mesh.use_auto_smooth = True
        mesh.show_edge_sharp = True
        return False
    elif fbx_layer_mapping == b'ByPolygon':
        blen_data = mesh.polygons
        return blen_read_geom_array_mapped_polygon(
            mesh, blen_data, "use_smooth",
            fbx_layer_data, None,
            fbx_layer_mapping, fbx_layer_ref,
            1, 1, layer_id,
            xform=lambda s: (s != 0),  # smoothgroup bitflags, treat as booleans for now
            )
    else:
        print("warning layer %r mapping type unsupported: %r" % (fbx_layer.id, fbx_layer_mapping))
        return False

def blen_read_geom_layer_material(fbxSDKImport, mesh_index, mesh):
    indice_size = fbxSDKImport.get_mesh_mat_indice_size(mesh_index)
    
    if indice_size == 0:
        return
    
    indices = (c_int * indice_size)()
    fbx_layerInfo = LayerElementInfo()
    fbxSDKImport.get_mesh_material_info(mesh_index, byref(indices), len(indices), byref(fbx_layerInfo))
    fbx_layer_data = array.array('l', ())
    for index in indices:
        fbx_layer_data.append(index)    
    
    fbx_layer_name = ""
    fbx_layer_mapping = fbx_layerInfo.MappingType
    fbx_layer_ref = fbx_layerInfo.RefType
    layer_id = b'Materials'
    
    blen_data = mesh.polygons
    blen_read_geom_array_mapped_polygon(
        mesh, blen_data, "material_index",
        fbx_layer_data, None,
        fbx_layer_mapping, fbx_layer_ref,
        1, 1, layer_id,
        )
    
def blen_read_material(fbxSDKImport, settings, material_index):
    elem_name_utf8 = fbxSDKImport.get_material_name(material_index).decode('utf-8')
    const_color_white = 1.0, 1.0, 1.0
    cycles_material_wrap_map = settings.cycles_material_wrap_map
    ma = bpy.data.materials.new(name=elem_name_utf8)    
    
    emissive = Vector3(0.0, 0.0, 0.0)
    ambient = Vector3(0.0, 0.0, 0.0)
    diffuse = Vector3(0.0, 0.0, 0.0)
    fbxSDKImport.get_material_props(material_index, byref(emissive), byref(ambient), byref(diffuse))
    
    ma_diff = [diffuse.x, diffuse.y, diffuse.z]
    ma_spec = const_color_white
    ma_alpha = 1.0
    ma_spec_intensity = ma.specular_intensity = 0.25 * 2.0
    ma_spec_hardness = 9.6
    ma_refl_factor = 0.0
    ma_refl_color = const_color_white
    
    if settings.use_cycles:
        from modules import cycles_shader_compat
        # viewport color
        ma.diffuse_color = ma_diff

        ma_wrap = cycles_shader_compat.CyclesShaderWrapper(ma)
        ma_wrap.diffuse_color_set(ma_diff)
        ma_wrap.specular_color_set([c * ma_spec_intensity for c in ma_spec])
        ma_wrap.hardness_value_set(((ma_spec_hardness + 3.0) / 5.0) - 0.65)
        ma_wrap.alpha_value_set(ma_alpha)
        ma_wrap.reflect_factor_set(ma_refl_factor)
        ma_wrap.reflect_color_set(ma_refl_color)

        cycles_material_wrap_map[ma] = ma_wrap
    else:
        # TODO, number BumpFactor isnt used yet
        ma.diffuse_color = ma_diff
        ma.specular_color = ma_spec
        ma.alpha = ma_alpha
        if ma_alpha < 1.0:
            ma.use_transparency = True
            ma.transparency_method = 'RAYTRACE'
        ma.specular_intensity = ma_spec_intensity
        ma.specular_hardness = ma_spec_hardness * 5.10 + 1.0

        if ma_refl_factor != 0.0:
            ma.raytrace_mirror.use = True
            ma.raytrace_mirror.reflect_factor = ma_refl_factor
            ma.mirror_color = ma_refl_color
            
    fbx_obj = FBXElem(
        b'Material', elem_name_utf8, b"Material", None
    )
        
    return fbx_obj, ma                

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
        
        blen_read_geom_layer_material(fbxSDKImport, mesh_index, mesh)
        blen_read_geom_layer_uv(fbxSDKImport, mesh_index, mesh)
        
    edge_size = fbxSDKImport.get_mesh_edge_size(mesh_index)
    if edge_size > 0:
        edges = (c_int * (edge_size * 2))()
        fbxSDKImport.get_mesh_edges(mesh_index, byref(edges), len(edges))
        edges_conv = array.array('i', ())
        for e in edges:
            edges_conv.append(e) 
        mesh.edges.add(edge_size)
        mesh.edges.foreach_set("vertices", edges_conv)
        
    ok_smooth = blen_read_geom_layer_smooth(fbxSDKImport, mesh_index, mesh)
    
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
    
    if ok_normals:
        clnors = array.array('f', [0.0] * (len(mesh.loops) * 3))
        mesh.loops.foreach_get("normal", clnors)
        
        if not ok_smooth:
            mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
            ok_smooth = True
        
        mesh.normals_split_custom_set(tuple(zip(*(iter(clnors),) * 3)))
        mesh.use_auto_smooth = True
        mesh.show_edge_sharp = True
    else:
        mesh.calc_normals()

    if settings.use_custom_normals:
        mesh.free_normals_split()
        
    if not ok_smooth:
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))    
        
    fbx_obj = FBXElem(
        b'Geometry', elem_name_utf8, b"Geometry", None
    )
        
    return fbx_obj, mesh

def blen_read_texture_image(fbxSDKImport, settings, texture_index, basedir):
    import os
    from bpy_extras import image_utils
    
    imagepath = None
    print("blen_read_texture_image")
    elem_name_utf8 = fbxSDKImport.get_texture_name(texture_index).decode('utf-8')
    
    image_cache = settings.image_cache
    
    filepath = fbxSDKImport.get_texture_rel_filename(texture_index).decode('utf-8')
    if filepath:
        filepath = os.path.join(basedir, filepath)
        filepath = filepath.replace('\\', '/') if (os.sep == '/') else filepath.replace('/', '\\')
        filepath = bpy.path.native_pathsep(filepath)
        if os.path.exists(filepath):
            imagepath = filepath
    
    if imagepath is None:
        filepath = fbxSDKImport.get_texture_filename(texture_index).decode('utf-8')
        if filepath:
            filepath = filepath.replace('\\', '/') if (os.sep == '/') else filepath.replace('/', '\\')
            filepath = bpy.path.native_pathsep(filepath)
            if os.path.exists(filepath):
                imagepath = filepath
    
    if imagepath is None:
        print("Error, could not find any file path in ", texture_index)
        print("       Falling back to: ", elem_name_utf8)
        filepath = elem_name_utf8
        filepath = filepath.replace('\\', '/') if (os.sep == '/') else filepath.replace('/', '\\')
        imagepath = filepath
        
    print(imagepath)
        
    image = image_cache.get(filepath)
    if image is not None:
        # Data is only embedded once, we may have already created the image but still be missing its data!
        if not image.has_data:
            pack_data_from_content(image, fbx_obj)
        return image        

    image = image_utils.load_image(
        filepath,
        dirname=basedir,
        place_holder=True,
        recursive=settings.use_image_search,
        )
    
    image_cache[filepath] = image
    # name can be ../a/b/c
    image.name = os.path.basename(elem_name_utf8)
    
    fbx_obj = FBXElem(
        b'Texture', elem_name_utf8, b"Texture", None
    )    
    
    return fbx_obj, image    
    
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
    
    fbxSDKImport.print_mesh()
    
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        
    # deselect all
    if bpy.ops.object.select_all.poll():
        bpy.ops.object.select_all(action='DESELECT')
        
    basedir = os.path.dirname(filepath)
        
    cycles_material_wrap_map = {}
    image_cache = {}
    if not use_cycles:
        texture_cache = {}    
        
    # Tables: (FBX_byte_id -> [FBX_data, None or Blender_datablock])
    fbx_table_nodes = {}        
        
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
        global_matrix, global_scale, bake_space_transform, use_custom_normals, global_matrix_inv_transposed, global_matrix_inv,
        cycles_material_wrap_map, use_cycles, image_cache, use_image_search,
    )
    
    fbx_helper_nodes = {}
    # create scene root
    fbx_helper_nodes[0] = root_helper = FbxImportHelperNode(None, None, None, False)
    root_helper.is_root = True
    
    model_count = fbxSDKImport.get_model_count()
    for i in range(model_count):
        fbx_uuid = fbxSDKImport.get_model_uuid(i)
        fbx_obj, transform_data = blen_read_model_transform_preprocess(fbxSDKImport, i, Matrix(), use_prepost_rot)
        fbx_helper_nodes[fbx_uuid] = FbxImportHelperNode(fbx_obj, None, transform_data, False)
    
    mesh_count = fbxSDKImport.get_mesh_count()
    for i in range(mesh_count):
        fbx_uuid = fbxSDKImport.get_mesh_uuid(i)
        fbx_obj, bl_data = blen_read_geom(fbxSDKImport, settings, i)
        fbx_table_nodes[fbx_uuid] = [fbx_obj, bl_data]
        
    material_count = fbxSDKImport.get_material_count()
    for i in range(material_count):
        fbx_uuid = fbxSDKImport.get_material_uuid(i)
        fbx_obj, bl_data = blen_read_material(fbxSDKImport, settings, i)
        fbx_table_nodes[fbx_uuid] = [fbx_obj, bl_data]
        
    texture_count = fbxSDKImport.get_texture_count()
    for i in range(texture_count):
        fbx_uuid = fbxSDKImport.get_texture_uuid(i)
        fbx_obj, bl_data = blen_read_texture_image(fbxSDKImport, settings, i, basedir)
        fbx_table_nodes[fbx_uuid] = [fbx_obj, bl_data]
        
    connection_size = fbxSDKImport.get_connection_count()
    connections = (UInt64Vector2 * connection_size)()
    fbxSDKImport.get_connections(byref(connections), len(connections))
    print("connections")
    for c in connections:
        print("%d, %d" % (c.x, c.y))
        c_dst = c.x
        c_src = c.y
        parent = fbx_helper_nodes.get(c_dst)
        if parent is None:
            continue
        
        child = fbx_helper_nodes.get(c_src)
        if child is None:
            # add blender data (meshes, lights, cameras, etc.) to a helper node
            fbx_sdata, bl_data = p_item = fbx_table_nodes.get(c_src, (None, None))
            if fbx_sdata is None:
                continue
            if fbx_sdata.id not in {b'Geometry'}:
                continue
            parent.bl_data = bl_data
        else:
            # set parent
            child.parent = parent
    
    '''    
    parent = fbx_helper_nodes.get(0)
    child = fbx_helper_nodes.get(1)
    child.parent = parent
    '''
              
    root_helper.find_correction_matrix(settings)
    root_helper.build_hierarchy(settings, scene)
    root_helper.link_hierarchy(settings, scene)
    root_helper.print_info(0)

    return {'FINISHED'}
    
    
        