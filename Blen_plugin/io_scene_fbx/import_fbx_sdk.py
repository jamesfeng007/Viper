import ctypes
from ctypes import c_char_p, byref, POINTER, c_double, c_int, c_char
import array
import bpy
from mathutils import Matrix, Euler, Vector
from . import fbx_utils, data_types
from fbx_ie_lib import FBXImport, GlobalSettings, ObjectTransformProp, LayerElementInfo, Vector3, UInt64Vector2, IntVector2
from collections import namedtuple
from .fbx_utils import (
    units_blender_to_fbx_factor,
    FBX_FRAMERATES,
    units_convertor_iter,
    similar_values_iter,
    array_to_matrix4,
)

if "bpy" in locals():
    import importlib
    if "fbx_utils" in locals():
        importlib.reload(fbx_utils)
    if "data_types" in locals():
        importlib.reload(data_types)        
        
FBXImportSettings = namedtuple("FBXImportSettings", (
    "global_matrix", "global_scale", "bake_space_transform", "use_custom_normals", "global_matrix_inv_transposed", "global_matrix_inv",
    "cycles_material_wrap_map", "use_cycles", "image_cache", "use_image_search", "ignore_leaf_bones", "automatic_bone_orientation",
    "bone_correction_matrix", "force_connect_children",
))

FBXTransformData = namedtuple("FBXTransformData", (
    "loc", "geom_loc",
    "rot", "rot_ofs", "rot_piv", "pre_rot", "pst_rot", "rot_ord", "rot_alt_mat", "geom_rot",
    "sca", "sca_ofs", "sca_piv", "geom_sca",
))

FBXElem = namedtuple("FBXElem", ("id", "props", "props_type", "elems"))

convert_deg_to_rad_iter = units_convertor_iter("degree", "radian")

# XXX This might be weak, now that we can add vgroups from both bones and shapes, name collisions become
#     more likely, will have to make this more robust!!!
def add_vgroup_to_objects(vg_indices, vg_weights, vg_name, objects):
    assert(len(vg_indices) == len(vg_weights))
    if vg_indices:
        for obj in objects:
            # We replace/override here...
            vg = obj.vertex_groups.get(vg_name)
            if vg is None:
                vg = obj.vertex_groups.new(vg_name)
            for i, w in zip(vg_indices, vg_weights):
                vg.add((i,), w, 'REPLACE')

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
            
    def get_world_matrix_as_parent(self):
        matrix = self.parent.get_world_matrix_as_parent() if self.parent else Matrix()
        if self.matrix_as_parent:
            matrix = matrix * self.matrix_as_parent
        return matrix            
            
    def get_world_matrix(self):
        matrix = self.parent.get_world_matrix_as_parent() if self.parent else Matrix()
        if self.matrix:
            matrix = matrix * self.matrix
        return matrix            
            
    def get_matrix(self):
        matrix = self.matrix if self.matrix else Matrix()
        if self.pre_matrix:
            matrix = self.pre_matrix * matrix
        if self.post_matrix:
            matrix = matrix * self.post_matrix
        return matrix
    
    def get_bind_matrix(self):
        matrix = self.bind_matrix if self.bind_matrix else Matrix()
        if self.pre_matrix:
            matrix = self.pre_matrix * matrix
        if self.post_matrix:
            matrix = matrix * self.post_matrix
        return matrix
    
    def make_bind_pose_local(self, parent_matrix=None):
        if parent_matrix is None:
            parent_matrix = Matrix()

        if self.bind_matrix:
            bind_matrix = parent_matrix.inverted_safe() * self.bind_matrix
        else:
            bind_matrix = self.matrix.copy() if self.matrix else None

        self.bind_matrix = bind_matrix
        if bind_matrix:
            parent_matrix = parent_matrix * bind_matrix

        for child in self.children:
            child.make_bind_pose_local(parent_matrix)        
        
    def mark_leaf_bones(self):
        if self.is_bone and len(self.children) == 1:
            child = self.children[0]
            if child.is_bone and len(child.children) == 0:
                child.is_leaf = True
        for child in self.children:
            child.mark_leaf_bones()        
        
    def do_bake_transform(self, settings):
        return (settings.bake_space_transform and self.fbx_type in (b'Mesh', b'Null') and
                not self.is_armature and not self.is_bone)        
        
    def find_correction_matrix(self, settings, parent_correction_inv=None):
        if self.parent and (self.parent.is_root or self.parent.do_bake_transform(settings)):
            self.pre_matrix = settings.global_matrix

        if parent_correction_inv:
            self.pre_matrix = parent_correction_inv * (self.pre_matrix if self.pre_matrix else Matrix())

        correction_matrix = None
        
        if self.is_bone:
            if settings.automatic_bone_orientation:
                pass
            else:
                correction_matrix = settings.bone_correction_matrix

        self.post_matrix = correction_matrix

        if self.do_bake_transform(settings):
            self.post_matrix = settings.global_matrix_inv * (self.post_matrix if self.post_matrix else Matrix())

        # process children
        correction_matrix_inv = correction_matrix.inverted_safe() if correction_matrix else None
        for child in self.children:
            child.find_correction_matrix(settings, correction_matrix_inv)
            
    def find_armature_bones(self, armature):
        for child in self.children:
            if child.is_bone:
                child.armature = armature
                child.find_armature_bones(armature)            
            
    def find_armatures(self):
        needs_armature = False
        for child in self.children:
            if child.is_bone:
                needs_armature = True
                break
        if needs_armature:
            if self.fbx_type in {b'Null', b'Root'}:
                # if empty then convert into armature
                self.is_armature = True
                armature = self
            else:
                # otherwise insert a new node
                # XXX Maybe in case self is virtual FBX root node, we should instead add one armature per bone child?
                armature = FbxImportHelperNode(None, None, None, False)
                armature.fbx_name = "Armature"
                armature.is_armature = True

                for child in tuple(self.children):
                    if child.is_bone:
                        child.parent = armature

                armature.parent = self

            armature.find_armature_bones(armature)

        for child in self.children:
            if child.is_armature or child.is_bone:
                continue
            child.find_armatures()
            
    def find_bone_children(self):
        has_bone_children = False
        for child in self.children:
            has_bone_children |= child.find_bone_children()
        self.has_bone_children = has_bone_children
        return self.is_bone or has_bone_children
    
    def find_fake_bones(self, in_armature=False):
        if in_armature and not self.is_bone and self.has_bone_children:
            self.is_bone = True
            # if we are not a null node we need an intermediate node for the data
            if self.fbx_type not in {b'Null', b'Root'}:
                node = FbxImportHelperNode(self.fbx_elem, self.bl_data, None, False)
                self.fbx_elem = None
                self.bl_data = None

                # transfer children
                for child in self.children:
                    if child.is_bone or child.has_bone_children:
                        continue
                    child.parent = node

                # attach to parent
                node.parent = self

        if self.is_armature:
            in_armature = True
        for child in self.children:
            child.find_fake_bones(in_armature)
            
    def collect_skeleton_meshes(self, meshes):
        for _, m in self.clusters:
            meshes.update(m)
        for child in self.children:
            child.collect_skeleton_meshes(meshes)            
            
    def collect_armature_meshes(self):
        if self.is_armature:
            armature_matrix_inv = self.get_world_matrix().inverted_safe()

            meshes = set()
            for child in self.children:
                child.collect_skeleton_meshes(meshes)
            for m in meshes:
                old_matrix = m.matrix
                m.matrix = armature_matrix_inv * m.get_world_matrix()
                m.anim_compensation_matrix = old_matrix.inverted_safe() * m.matrix
                m.is_global_animation = True
                m.parent = self
            self.meshes = meshes
        else:
            for child in self.children:
                child.collect_armature_meshes()            
            
    def build_skeleton(self, arm, parent_matrix, parent_bone_size=1, force_connect_children=False):
        def child_connect(par_bone, child_bone, child_head, connect_ctx):
            # child_bone or child_head may be None.
            force_connect_children, connected = connect_ctx
            if child_bone is not None:
                child_bone.parent = par_bone
                child_head = child_bone.head

            if similar_values_iter(par_bone.tail, child_head):
                if child_bone is not None:
                    child_bone.use_connect = True
                # Disallow any force-connection at this level from now on, since that child was 'really'
                # connected, we do not want to move current bone's tail anymore!
                connected = None
            elif force_connect_children and connected is not None:
                # We only store position where tail of par_bone should be in the end.
                # Actual tail moving and force connection of compatible child bones will happen
                # once all have been checked.
                if connected is ...:
                    connected = ([child_head.copy(), 1], [child_bone] if child_bone is not None else [])
                else:
                    connected[0][0] += child_head
                    connected[0][1] += 1
                    if child_bone is not None:
                        connected[1].append(child_bone)
            connect_ctx[1] = connected

        def child_connect_finalize(par_bone, connect_ctx):
            force_connect_children, connected = connect_ctx
            # Do nothing if force connection is not enabled!
            if force_connect_children and connected is not None and connected is not ...:
                # Here again we have to be wary about zero-length bones!!!
                par_tail = connected[0][0] / connected[0][1]
                if (par_tail - par_bone.head).magnitude < 1e-2:
                    par_bone_vec = (par_bone.tail - par_bone.head).normalized()
                    par_tail = par_bone.head + par_bone_vec * 0.01
                par_bone.tail = par_tail
                for child_bone in connected[1]:
                    if similar_values_iter(par_tail, child_bone.head):
                        child_bone.use_connect = True

        # Create the (edit)bone.
        bone = arm.bl_data.edit_bones.new(name=self.fbx_name)
        bone.select = True
        self.bl_obj = arm.bl_obj
        self.bl_data = arm.bl_data
        self.bl_bone = bone.name  # Could be different from the FBX name!

        # get average distance to children
        bone_size = 0.0
        bone_count = 0
        for child in self.children:
            if child.is_bone:
                bone_size += child.get_bind_matrix().to_translation().magnitude
                bone_count += 1
        if bone_count > 0:
            bone_size /= bone_count
        else:
            bone_size = parent_bone_size

        # So that our bone gets its final length, but still Y-aligned in armature space.
        # 0-length bones are automatically collapsed into their parent when you leave edit mode,
        # so this enforces a minimum length.
        bone_tail = Vector((0.0, 1.0, 0.0)) * max(0.01, bone_size)
        bone.tail = bone_tail

        # And rotate/move it to its final "rest pose".
        bone_matrix = parent_matrix * self.get_bind_matrix().normalized()

        bone.matrix = bone_matrix

        # Correction for children attached to a bone. FBX expects to attach to the head of a bone,
        # while Blender attaches to the tail.
        self.bone_child_matrix = Matrix.Translation(-bone_tail)

        connect_ctx = [force_connect_children, ...]
        for child in self.children:
            if child.is_leaf and force_connect_children:
                # Arggggggggggggggggg! We do not want to create this bone, but we need its 'virtual head' location
                # to orient current one!!!
                child_head = (bone_matrix * child.get_bind_matrix().normalized()).translation
                child_connect(bone, None, child_head, connect_ctx)
            elif child.is_bone and not child.ignore:
                child_bone = child.build_skeleton(arm, bone_matrix, bone_size,
                                                  force_connect_children=force_connect_children)
                # Connection to parent.
                child_connect(bone, child_bone, None, connect_ctx)

        child_connect_finalize(bone, connect_ctx)
        return bone            
            
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
    
    def build_skeleton_children(self, settings, scene):
        if self.is_bone:
            for child in self.children:
                if child.ignore:
                    continue
                child.build_skeleton_children(settings, scene)
            return None
        else:
            # child is not a bone
            obj = self.build_node_obj(settings)

            if obj is None:
                return None

            for child in self.children:
                if child.ignore:
                    continue
                child.build_skeleton_children(settings, scene)

            # instance in scene
            obj_base = scene.objects.link(obj)
            obj_base.select = True

            return obj
        
    def link_skeleton_children(self, settings, scene):
        if self.is_bone:
            for child in self.children:
                if child.ignore:
                    continue
                child_obj = child.bl_obj
                if child_obj and child_obj != self.bl_obj:
                    child_obj.parent = self.bl_obj  # get the armature the bone belongs to
                    child_obj.parent_bone = self.bl_bone
                    child_obj.parent_type = 'BONE'
                    child_obj.matrix_parent_inverse = Matrix()

                    # Blender attaches to the end of a bone, while FBX attaches to the start.
                    # bone_child_matrix corrects for that.
                    if child.pre_matrix:
                        child.pre_matrix = self.bone_child_matrix * child.pre_matrix
                    else:
                        child.pre_matrix = self.bone_child_matrix

                    child_obj.matrix_basis = child.get_matrix()
            return None
        else:
            obj = self.bl_obj

            for child in self.children:
                if child.ignore:
                    continue
                child_obj = child.link_skeleton_children(settings, scene)
                if child_obj:
                    child_obj.parent = obj

            return obj        
    
    def set_pose_matrix(self, arm):
        pose_bone = arm.bl_obj.pose.bones[self.bl_bone]
        pose_bone.matrix_basis = self.get_bind_matrix().inverted_safe() * self.get_matrix()

        for child in self.children:
            if child.ignore:
                continue
            if child.is_bone:
                child.set_pose_matrix(arm)
                
    def merge_weights(self, combined_weights, fbx_cluster):
        indices = fbx_cluster.elems[0]
        weights = fbx_cluster.elems[1]

        for index, weight in zip(indices, weights):
            w = combined_weights.get(index)
            if w is None:
                combined_weights[index] = [weight]
            else:
                w.append(weight)                
                
    def set_bone_weights(self):
        ignored_children = tuple(child for child in self.children
                                       if child.is_bone and child.ignore and len(child.clusters) > 0)

        if len(ignored_children) > 0:
            # If we have an ignored child bone we need to merge their weights into the current bone weights.
            # This can happen both intentionally and accidentally when skinning a model. Either way, they
            # need to be moved into a parent bone or they cause animation glitches.
            for fbx_cluster, meshes in self.clusters:
                combined_weights = {}
                self.merge_weights(combined_weights, fbx_cluster)

                for child in ignored_children:
                    for child_cluster, child_meshes in child.clusters:
                        if not meshes.isdisjoint(child_meshes):
                            self.merge_weights(combined_weights, child_cluster)

                # combine child weights
                indices = []
                weights = []
                for i, w in combined_weights.items():
                    indices.append(i)
                    if len(w) > 1:
                        weights.append(sum(w) / len(w))
                    else:
                        weights.append(w[0])

                add_vgroup_to_objects(indices, weights, self.bl_bone, [node.bl_obj for node in meshes])

            # clusters that drive meshes not included in a parent don't need to be merged
            all_meshes = set().union(*[meshes for _, meshes in self.clusters])
            for child in ignored_children:
                for child_cluster, child_meshes in child.clusters:
                    if all_meshes.isdisjoint(child_meshes):
                        indices = child_cluster.elems[0]
                        weights = child_cluster.elems[1]
                        add_vgroup_to_objects(indices, weights, self.bl_bone, [node.bl_obj for node in child_meshes])
        else:
            # set the vertex weights on meshes
            for fbx_cluster, meshes in self.clusters:
                indices = fbx_cluster.elems[0]
                weights = fbx_cluster.elems[1]        
                add_vgroup_to_objects(indices, weights, self.bl_bone, [node.bl_obj for node in meshes])

        for child in self.children:
            if child.is_bone and not child.ignore:
                child.set_bone_weights()                
            
    def build_hierarchy(self, settings, scene):
        if self.is_armature:
            # create when linking since we need object data
            elem_name_utf8 = self.fbx_name

            self.bl_data = arm_data = bpy.data.armatures.new(name=elem_name_utf8)

            # Object data must be created already
            self.bl_obj = arm = bpy.data.objects.new(name=elem_name_utf8, object_data=arm_data)

            arm.matrix_basis = self.get_matrix()
            
            #if self.fbx_elem:
                #if settings.use_custom_props:
                    #blen_read_custom_properties(self.fbx_elem, arm, settings)
                    
            # instance in scene
            obj_base = scene.objects.link(arm)
            obj_base.select = True

            # Add bones:

            # Switch to Edit mode.
            scene.objects.active = arm
            is_hidden = arm.hide
            arm.hide = False  # Can't switch to Edit mode hidden objects...
            bpy.ops.object.mode_set(mode='EDIT')

            for child in self.children:
                if child.ignore:
                    continue
                if child.is_bone:
                    child.build_skeleton(self, Matrix(), force_connect_children=settings.force_connect_children)

            bpy.ops.object.mode_set(mode='OBJECT')

            arm.hide = is_hidden

            # Set pose matrix
            for child in self.children:
                if child.ignore:
                    continue
                if child.is_bone:
                    child.set_pose_matrix(self)

            # Add bone children:
            for child in self.children:
                if child.ignore:
                    continue
                child_obj = child.build_skeleton_children(settings, scene)

            return arm                    
            
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
            arm = self.bl_obj

            # Link bone children:
            for child in self.children:
                if child.ignore:
                    continue
                child_obj = child.link_skeleton_children(settings, scene)
                if child_obj:
                    child_obj.parent = arm
                    
            # Add armature modifiers to the meshes
            if self.meshes:
                for mesh in self.meshes:
                    (mmat, amat) = mesh.armature_setup[self]
                    me_obj = mesh.bl_obj

                    # bring global armature & mesh matrices into *Blender* global space.
                    # Note: Usage of matrix_geom (local 'diff' transform) here is quite brittle.
                    #       Among other things, why in hell isn't it taken into account by bindpose & co???
                    #       Probably because org app (max) handles it completely aside from any parenting stuff,
                    #       which we obviously cannot do in Blender. :/
                    if amat is None:
                        amat = self.bind_matrix
                    amat = settings.global_matrix * (Matrix() if amat is None else amat)
                    if self.matrix_geom:
                        amat = amat * self.matrix_geom
                    mmat = settings.global_matrix * mmat
                    if mesh.matrix_geom:
                        mmat = mmat * mesh.matrix_geom

                    # Now that we have armature and mesh in there (global) bind 'state' (matrix),
                    # we can compute inverse parenting matrix of the mesh.
                    me_obj.matrix_parent_inverse = amat.inverted_safe() * mmat * me_obj.matrix_basis.inverted_safe()

                    mod = mesh.bl_obj.modifiers.new(arm.name, 'ARMATURE')
                    mod.object = arm

            # Add bone weights to the deformers
            for child in self.children:
                if child.ignore:
                    continue
                if child.is_bone:
                    child.set_bone_weights()

            return arm                                
            
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

def blen_read_bone(fbxSDKImport, bone_index):
    elem_name_utf8 = fbxSDKImport.get_bone_name(bone_index).decode('utf-8')
    
    fbx_obj = FBXElem(
        b'NodeAttribute', elem_name_utf8, b"NodeAttribute", None
    )
    
    return fbx_obj, None
    
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

def blen_read_cluster(fbxSDKImport, cluster_index):
    elem_name_utf8 = fbxSDKImport.get_cluster_name(cluster_index).decode('utf-8')
    
    indice_size = fbxSDKImport.get_cluster_indice_size(cluster_index)
    if indice_size > 0:        
        indice = (c_int * indice_size)()
        weights = (c_double * indice_size)()
        fbxSDKImport.get_cluster_weight_indice(cluster_index, byref(indice), byref(weights), indice_size)
        fbx_weights = array.array(data_types.ARRAY_FLOAT64, ())
        fbx_indice = array.array('l', ())
        for i in range(indice_size):
            fbx_weights.append(weights[i])
            fbx_indice.append(indice[i])
    else:
        fbx_weights = ()
        fbx_indice = ()
    
    fbx_transform = (c_double * 16)()
    fbx_link_transform = (c_double * 16)()
    fbxSDKImport.get_cluster_transforms(cluster_index, byref(fbx_transform), byref(fbx_link_transform), 16)
    
    arr_transform = array.array(data_types.ARRAY_FLOAT64, ())
    arr_transform_link = array.array(data_types.ARRAY_FLOAT64, ())
    for i in range(16):
        arr_transform.append(fbx_transform[i])
        arr_transform_link.append(fbx_link_transform[i])
        
    transform_matrix = array_to_matrix4(arr_transform)
    transform_link_matrix = array_to_matrix4(arr_transform_link)
    
    cluster = (fbx_indice, fbx_weights, transform_matrix, transform_link_matrix)
    
    fbx_obj = FBXElem(
        b'Deformer', elem_name_utf8, b"Cluster", cluster
    )
    
    return fbx_obj, cluster

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
        b'Geometry', elem_name_utf8, b"Mesh", None
    )
        
    return fbx_obj, mesh

def blen_read_texture_image(fbxSDKImport, settings, texture_index, basedir):
    import os
    from bpy_extras import image_utils
    
    imagepath = None
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
    
    mat_prop = fbxSDKImport.get_texture_mat_prop(texture_index)
    translation = Vector3(0.0, 0.0, 0.0)
    rotation = Vector3(0.0, 0.0, 0.0)
    scaling = Vector3(0.0, 0.0, 0.0)
    wrap_mode = IntVector2(0, 0)
    fbxSDKImport.get_texture_mapping(texture_index, byref(translation), byref(rotation), byref(scaling), byref(wrap_mode))
    
    fbx_obj = FBXElem(
        b'Texture', elem_name_utf8, b"Texture", (mat_prop, ((translation.x, translation.y, translation.z), (rotation.x, rotation.y, rotation.z), 
                                                 (scaling.x, scaling.y, scaling.z), (bool(wrap_mode.x), bool(wrap_mode.y))))
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
    fbxSDKImport.print_node()
    fbxSDKImport.print_skeleton()
    
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
        cycles_material_wrap_map, use_cycles, image_cache, use_image_search, ignore_leaf_bones, automatic_bone_orientation,
        bone_correction_matrix, force_connect_children,
    )
    
    fbx_helper_nodes = {}
    # create scene root
    fbx_helper_nodes[0] = root_helper = FbxImportHelperNode(None, None, None, False)
    root_helper.is_root = True
    
    print("FBX import: Nodes...")
    
    model_count = fbxSDKImport.get_model_count()
    for i in range(model_count):
        fbx_uuid = fbxSDKImport.get_model_uuid(i)
        fbx_obj, transform_data = blen_read_model_transform_preprocess(fbxSDKImport, i, Matrix(), use_prepost_rot)
        is_bone = fbxSDKImport.is_model_bone(i)
        fbx_helper_nodes[fbx_uuid] = FbxImportHelperNode(fbx_obj, None, transform_data, is_bone) 
        
    print("FBX import: Meshes & Clusters & Skin...")
    
    mesh_count = fbxSDKImport.get_mesh_count()
    for i in range(mesh_count):
        fbx_uuid = fbxSDKImport.get_mesh_uuid(i)
        fbx_obj, bl_data = blen_read_geom(fbxSDKImport, settings, i)
        fbx_table_nodes[fbx_uuid] = [fbx_obj, bl_data]
        
    cluster_count = fbxSDKImport.get_cluster_count()
    for i in range(cluster_count):
        fbx_uuid = fbxSDKImport.get_cluster_uuid(i)
        fbx_obj, bl_data = blen_read_cluster(fbxSDKImport, i)
        fbx_table_nodes[fbx_uuid] = [fbx_obj, bl_data]
        
    skin_count = fbxSDKImport.get_skin_count()
    for i in range(skin_count):
        fbx_uuid = fbxSDKImport.get_skin_uuid(i)
        elem_name_utf8 = fbxSDKImport.get_skin_name(i).decode('utf-8')
        fbx_obj = FBXElem(
            b'Deformer', elem_name_utf8, b"Skin", None
        )
        fbx_table_nodes[fbx_uuid] = [fbx_obj, None]
        
    print("FBX import: Materials & Textures...")
        
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
        
    print("FBX import: Bones...")
    
    bone_count = fbxSDKImport.get_bone_count()
    for i in range(bone_count):
        fbx_uuid = fbxSDKImport.get_bone_uuid(i)
        fbx_obj, bl_data = blen_read_bone(fbxSDKImport, i)
        fbx_table_nodes[fbx_uuid] = [fbx_obj, bl_data]
        
    print("FBX import: Connections...")
        
    connection_size = fbxSDKImport.get_connection_count()
    connections = (UInt64Vector2 * connection_size)()
    fbxSDKImport.get_connections(byref(connections), len(connections))
    
    print("FBX import: Objects & Armatures...")
    
    for c in connections:
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
            if fbx_sdata.id not in {b'Geometry', b'NodeAttribute'}:
                continue
            parent.bl_data = bl_data
        else:
            # set parent
            child.parent = parent
              
    # find armatures (either an empty below a bone or a new node inserted at the bone
    root_helper.find_armatures()
    
    # mark nodes that have bone children
    root_helper.find_bone_children()
    
    # mark nodes that need a bone to attach child-bones to
    root_helper.find_fake_bones()
    
    # mark leaf nodes that are only required to mark the end of their parent bone
    if settings.ignore_leaf_bones:
        root_helper.mark_leaf_bones()
    
    # get the bind pose from pose elements
    pose_count = fbxSDKImport.get_pose_count()
    for i in range(pose_count):
        fbx_ref_uuid = fbxSDKImport.get_ref_bone_uuid(i)
        fbx_matrix = (c_double * 16)()
        fbxSDKImport.get_pose_matrix(i, byref(fbx_matrix), len(fbx_matrix))
        arr_matrix = array.array(data_types.ARRAY_FLOAT64, ())
        for v in fbx_matrix:
            arr_matrix.append(v)
        matrix = array_to_matrix4(arr_matrix)
        bone = fbx_helper_nodes.get(fbx_ref_uuid)
        if bone and matrix:
            # Store the matrix in the helper node.
            # There may be several bind pose matrices for the same node, but in tests they seem to be identical.
            bone.bind_matrix = matrix  # global space
            
    # get clusters and bind pose
    for helper_uuid, helper_node in fbx_helper_nodes.items():
        if not helper_node.is_bone:
            continue
        for c in connections:
            c_dst = c.x
            c_src = c.y
            if helper_uuid == c_src:    #find cluster from bone
                cluster_uuid = c_dst
                fbx_cluster, bl_data = fbx_table_nodes.get(cluster_uuid, (None, None))
                if fbx_cluster is None or fbx_cluster.id != b'Deformer' or fbx_cluster.props_type != b'Cluster':
                    continue
                
                tx_mesh = bl_data[2]    #transform matrix
                tx_bone = bl_data[3]    #transform link matrix
                tx_arm = None
                
                mesh_matrix = tx_mesh
                armature_matrix = tx_arm                
                
                if tx_bone:
                    #mesh_matrix = tx_bone * mesh_matrix #already transformed by sdk
                    helper_node.bind_matrix = tx_bone  # overwrite the bind matrix
                    
                # Get the meshes driven by this cluster: (Shouldn't that be only one?)
                meshes = set()
                for _c in connections:
                    _c_dst = _c.x
                    _c_src = _c.y
                    if _c_src == cluster_uuid:   #from cluster to deformer
                        skin_uuid = _c_dst
                        fbx_skin, _ = fbx_table_nodes.get(skin_uuid, (None, None))
                        if fbx_skin is None or fbx_skin.id != b'Deformer' or fbx_skin.props_type != b'Skin':
                            continue
                        for __c in connections:
                            __c_dst = __c.x
                            __c_src = __c.y
                            if __c_src == skin_uuid: #from deformer to geometry
                                mesh_uuid = __c_dst
                                fbx_mesh, _ = fbx_table_nodes.get(mesh_uuid, (None, None))
                                if fbx_mesh is None or fbx_mesh.id != b'Geometry' or fbx_mesh.props_type != b'Mesh':
                                    continue
                                for ___c in connections:
                                    ___c_dst = ___c.x
                                    ___c_src = ___c.y
                                    if ___c_src == mesh_uuid: #from geometry to object node
                                        object_uuid = ___c_dst
                                        mesh_node = fbx_helper_nodes[object_uuid]
                                        if mesh_node:
                                            # ----
                                            # If we get a valid mesh matrix (in bone space), store armature and
                                            # mesh global matrices, we need them to compute mesh's matrix_parent_inverse
                                            # when actually binding them via the modifier.
                                            # Note we assume all bones were bound with the same mesh/armature (global) matrix,
                                            # we do not support otherwise in Blender anyway!
                                            mesh_node.armature_setup[helper_node.armature] = (mesh_matrix, armature_matrix)
                                            meshes.add(mesh_node)
                                            
                helper_node.clusters.append((fbx_cluster, meshes))
                            
    # convert bind poses from global space into local space
    root_helper.make_bind_pose_local()
    
    # collect armature meshes
    root_helper.collect_armature_meshes()    
    
    # find the correction matrices to align FBX objects with their Blender equivalent
    root_helper.find_correction_matrix(settings)
    
    # build the Object/Armature/Bone hierarchy
    root_helper.build_hierarchy(settings, scene)
    
    # Link the Object/Armature/Bone hierarchy
    root_helper.link_hierarchy(settings, scene)
    
    #root_helper.print_info(0)
    
    print("FBX import: Assign materials...")
    # link Material's to Geometry (via Model's)
    for fbx_uuid, fbx_item in fbx_table_nodes.items():
        fbx_obj, blen_data = fbx_item
        if fbx_obj.id != b'Geometry':
            continue

        mesh = fbx_table_nodes.get(fbx_uuid, (None, None))[1]   #found Mesh

        # can happen in rare cases
        if mesh is None:
            continue
        
        done_mats = set()
        
        for c in connections:
            c_parent = c.x
            c_child = c.y
            if fbx_uuid == c_child:
                fbx_lnk_uuid = c_parent     #found Model
                
                for c_lnk in connections:
                    c_lnk_parent = c_lnk.x
                    c_lnk_child = c_lnk.y
                    if fbx_lnk_uuid == c_lnk_parent and fbx_table_nodes.get(c_lnk_child, (None, None))[0].id == b'Material':
                        material = fbx_table_nodes.get(c_lnk_child, (None, None))[1]
                        
                        if material not in done_mats:
                            mesh.materials.append(material)
                            done_mats.add(material)                        
                
        # We have to validate mesh polygons' mat_idx, see T41015!
        # Some FBX seem to have an extra 'default' material which is not defined in FBX file.
        if mesh.validate_material_indices():
            print("WARNING: mesh '%s' had invalid material indices, those were reset to first material" % mesh.name)                
                
    print("FBX import: Assign textures...")
    
    if not use_cycles:
        # Simple function to make a new mtex and set defaults
        def material_mtex_new(material, image, tex_map):
            tex = texture_cache.get(image)
            if tex is None:
                tex = bpy.data.textures.new(name=image.name, type='IMAGE')
                tex.image = image
                texture_cache[image] = tex

                # copy custom properties from image object to texture
                for key, value in image.items():
                    tex[key] = value

                # delete custom properties on the image object
                for key in image.keys():
                    del image[key]

            mtex = material.texture_slots.add()
            mtex.texture = tex
            mtex.texture_coords = 'UV'
            mtex.use_map_color_diffuse = False

            # No rotation here...
            mtex.offset[:] = tex_map[0]
            mtex.scale[:] = tex_map[2]
            return mtex    
    
    material_images = {}
    for fbx_uuid, fbx_item in fbx_table_nodes.items():
        fbx_obj, blen_data = fbx_item
        if fbx_obj.id != b'Material':
            continue
        
        material = fbx_table_nodes.get(fbx_uuid, (None, None))[1]
        
        for c in connections:
            c_parent = c.x
            c_child = c.y
            if fbx_uuid == c_parent:
                fbx_lnk, image = fbx_table_nodes.get(c_child, (None, None))

                if use_cycles:
                    ma_wrap = cycles_material_wrap_map[material]
                    tex_map = fbx_lnk[3][1]
                    lnk_type = fbx_lnk[3][0]

                    if (tex_map[0] == (0.0, 0.0, 0.0) and
                            tex_map[1] == (0.0, 0.0, 0.0) and
                            tex_map[2] == (1.0, 1.0, 1.0) and
                            tex_map[3] == (False, False)):
                        use_mapping = False
                    else:
                        use_mapping = True
                        tex_map_kw = {
                            "translation": tex_map[0],
                            "rotation": [-i for i in tex_map[1]],
                            "scale": [((1.0 / i) if i != 0.0 else 1.0) for i in tex_map[2]],
                            "clamp": tex_map[3],
                            }

                    if lnk_type in {b'DiffuseColor', b'3dsMax|maps|texmap_diffuse'}:
                        ma_wrap.diffuse_image_set(image)
                        if use_mapping:
                            ma_wrap.diffuse_mapping_set(**tex_map_kw)
                    elif lnk_type == b'SpecularColor':
                        ma_wrap.specular_image_set(image)
                        if use_mapping:
                            ma_wrap.specular_mapping_set(**tex_map_kw)
                    elif lnk_type in {b'ReflectionColor', b'3dsMax|maps|texmap_reflection'}:
                        ma_wrap.reflect_image_set(image)
                        if use_mapping:
                            ma_wrap.reflect_mapping_set(**tex_map_kw)
                    elif lnk_type == b'TransparentColor':  # alpha
                        ma_wrap.alpha_image_set(image)
                        if use_mapping:
                            ma_wrap.alpha_mapping_set(**tex_map_kw)
                        if use_alpha_decals:
                            material_decals.add(material)
                    elif lnk_type == b'DiffuseFactor':
                        pass  # TODO
                    elif lnk_type == b'ShininessExponent':
                        ma_wrap.hardness_image_set(image)
                        if use_mapping:
                            ma_wrap.hardness_mapping_set(**tex_map_kw)
                    # XXX, applications abuse bump!
                    elif lnk_type in {b'NormalMap', b'Bump', b'3dsMax|maps|texmap_bump'}:
                        ma_wrap.normal_image_set(image)
                        ma_wrap.normal_factor_set(texture_bumpfac_get(fbx_obj))
                        if use_mapping:
                            ma_wrap.normal_mapping_set(**tex_map_kw)
                        """
                    elif lnk_type == b'Bump':
                        ma_wrap.bump_image_set(image)
                        ma_wrap.bump_factor_set(texture_bumpfac_get(fbx_obj))
                        if use_mapping:
                            ma_wrap.bump_mapping_set(**tex_map_kw)
                        """
                    else:
                        print("WARNING: material link %r ignored" % lnk_type)

                    material_images.setdefault(material, {})[lnk_type] = (image, tex_map)

                else:
                    tex_map = fbx_lnk[3][1]
                    lnk_type = fbx_lnk[3][0]
                    mtex = material_mtex_new(material, image, tex_map)
                    
                    if lnk_type in {b'DiffuseColor', b'3dsMax|maps|texmap_diffuse'}:
                        mtex.use_map_color_diffuse = True
                        mtex.blend_type = 'MULTIPLY'
                    elif lnk_type == b'SpecularColor':
                        mtex.use_map_color_spec = True
                        mtex.blend_type = 'MULTIPLY'
                    elif lnk_type in {b'ReflectionColor', b'3dsMax|maps|texmap_reflection'}:
                        mtex.use_map_raymir = True
                    elif lnk_type == b'TransparentColor':  # alpha
                        material.use_transparency = True
                        material.transparency_method = 'RAYTRACE'
                        material.alpha = 0.0
                        mtex.use_map_alpha = True
                        mtex.alpha_factor = 1.0
                        if use_alpha_decals:
                            material_decals.add(material)
                    elif lnk_type == b'DiffuseFactor':
                        mtex.use_map_diffuse = True
                    elif lnk_type == b'ShininessExponent':
                        mtex.use_map_hardness = True
                    # XXX, applications abuse bump!
                    elif lnk_type in {b'NormalMap', b'Bump', b'3dsMax|maps|texmap_bump'}:
                        mtex.texture.use_normal_map = True  # not ideal!
                        mtex.use_map_normal = True
                        mtex.normal_factor = texture_bumpfac_get(fbx_obj)
                        """
                    elif lnk_type == b'Bump':
                        mtex.use_map_normal = True
                        mtex.normal_factor = texture_bumpfac_get(fbx_obj)
                        """
                    else:
                        print("WARNING: material link %r ignored" % lnk_type)

                    material_images.setdefault(material, {})[lnk_type] = (image, tex_map)
                    
    # Check if the diffuse image has an alpha channel,
    # if so, use the alpha channel.

    # Note: this could be made optional since images may have alpha but be entirely opaque
    for fbx_uuid, fbx_item in fbx_table_nodes.items():
        fbx_obj, blen_data = fbx_item
        if fbx_obj.id != b'Material':
            continue
        material = fbx_table_nodes.get(fbx_uuid, (None, None))[1]
        image, tex_map = material_images.get(material, {}).get(b'DiffuseColor', (None, None))
        # do we have alpha?
        if image and image.depth == 32:
            if use_alpha_decals:
                material_decals.add(material)

            if use_cycles:
                ma_wrap = cycles_material_wrap_map[material]
                if ma_wrap.node_bsdf_alpha.mute:
                    ma_wrap.alpha_image_set_from_diffuse()
            else:
                if not any((True for mtex in material.texture_slots if mtex and mtex.use_map_alpha)):
                    mtex = material_mtex_new(material, image, tex_map)

                    material.use_transparency = True
                    material.transparency_method = 'RAYTRACE'
                    material.alpha = 0.0
                    mtex.use_map_alpha = True
                    mtex.alpha_factor = 1.0

        # propagate mapping from diffuse to all other channels which have none defined.
        if use_cycles:
            ma_wrap = cycles_material_wrap_map[material]
            ma_wrap.mapping_set_from_diffuse()                    
                    

    return {'FINISHED'}
    
    
        