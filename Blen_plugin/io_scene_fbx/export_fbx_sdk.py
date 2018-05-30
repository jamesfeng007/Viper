import os
import time
import array
import math  # math.pi
from ctypes import c_char_p, byref
from collections import OrderedDict

import bpy
import bpy_extras
from mathutils import Vector, Matrix
from . import data_types
from .fbx_utils import ( ObjectWrapper, FBXExportSettingsMedia, get_blenderID_key, BLENDER_OBJECT_TYPES_MESHLIKE, 
                         BLENDER_OTHER_OBJECT_TYPES, )
from print_util import PrintHelper

from fbx_export import FBXExport, Vector3, Mat4x4, Vector4, ChannelType

def to_channel_type(tx_chan, index):
    if tx_chan == 'T':
        if index == 0:
            return 0
        elif index == 1:
            return 1
        elif index == 2:
            return 2
    elif tx_chan == 'R':
        if index == 0:
            return 3
        elif index == 1:
            return 4
        elif index == 2:
            return 5
    elif tx_chan == 'S':
        if index == 0:
            return 6
        elif index == 1:
            return 7
        elif index == 2:
            return 8        

def grouper_exact(it, chunk_size):
    """
    Grouper-like func, but returns exactly all elements from it:

    >>> for chunk in grouper_exact(range(10), 3): print(e)
    (0,1,2)
    (3,4,5)
    (6,7,8)
    (9,)

    About 2 times slower than simple zip(*[it] * 3), but does not need to convert iterator to sequence to be sure to
    get exactly all elements in it (i.e. get a last chunk that may be smaller than chunk_size).
    """
    import itertools
    i = itertools.zip_longest(*[iter(it)] * chunk_size, fillvalue=...)
    curr = next(i)
    for nxt in i:
        yield curr
        curr = nxt
    if ... in curr:
        yield curr[:curr.index(...)]
    else:
        yield curr

# I guess FBX uses degrees instead of radians (Arystan).
# Call this function just before writing to FBX.
# 180 / math.pi == 57.295779513
def tuple_rad_to_deg(eul):
    return eul[0] * 57.295779513, eul[1] * 57.295779513, eul[2] * 57.295779513

def action_bone_names(obj, action):
    from bpy.types import PoseBone

    names = set()
    path_resolve = obj.path_resolve

    for fcu in action.fcurves:
        try:
            prop = path_resolve(fcu.data_path, False)
        except:
            prop = None

        if prop is not None:
            data = prop.data
            if isinstance(data, PoseBone):
                names.add(data.name)

    return names

# Used to add the scene name into the filepath without using odd chars
sane_name_mapping_ob = {}
sane_name_mapping_ob_unique = set()
sane_name_mapping_mat = {}
sane_name_mapping_tex = {}
sane_name_mapping_take = {}
sane_name_mapping_group = {}

# Make sure reserved names are not used
sane_name_mapping_ob['Scene'] = 'Scene_'
sane_name_mapping_ob_unique.add('Scene_')


def increment_string(t):
    name = t
    num = ''
    while name and name[-1].isdigit():
        num = name[-1] + num
        name = name[:-1]
    if num:
        return '%s%d' % (name, int(num) + 1)
    else:
        return name + '_0'


# todo - Disallow the name 'Scene' - it will bugger things up.
def sane_name(data, dct, unique_set=None):
    #if not data: return None

    if type(data) == tuple:  # materials are paired up with images
        data, other = data
        use_other = True
    else:
        other = None
        use_other = False

    name = data.name if data else None
    orig_name = name

    if other:
        orig_name_other = other.name
        name = '%s #%s' % (name, orig_name_other)
    else:
        orig_name_other = None

    # dont cache, only ever call once for each data type now,
    # so as to avoid namespace collision between types - like with objects <-> bones
    #try:        return dct[name]
    #except:        pass

    if not name:
        name = 'unnamed'  # blank string, ASKING FOR TROUBLE!
    else:

        name = bpy.path.clean_name(name)  # use our own

    name_unique = dct.values() if unique_set is None else unique_set

    while name in name_unique:
        name = increment_string(name)

    if use_other:  # even if other is None - orig_name_other will be a string or None
        dct[orig_name, orig_name_other] = name
    else:
        dct[orig_name] = name

    if unique_set is not None:
        unique_set.add(name)

    return name


def sane_obname(data):
    return sane_name(data, sane_name_mapping_ob, sane_name_mapping_ob_unique)


def sane_matname(data):
    return sane_name(data, sane_name_mapping_mat)


def sane_texname(data):
    return sane_name(data, sane_name_mapping_tex)


def sane_takename(data):
    return sane_name(data, sane_name_mapping_take)


def sane_groupname(data):
    return sane_name(data, sane_name_mapping_group)

# ob must be OB_MESH
def BPyMesh_meshWeight2List(ob, me):
    """ Takes a mesh and return its group names and a list of lists, one list per vertex.
    aligning the each vert list with the group names, each list contains float value for the weight.
    These 2 lists can be modified and then used with list2MeshWeight to apply the changes.
    """

    # Clear the vert group.
    groupNames = [g.name for g in ob.vertex_groups]
    len_groupNames = len(groupNames)

    if not len_groupNames:
        # no verts? return a vert aligned empty list
        return [[] for i in range(len(me.vertices))], []
    else:
        vWeightList = [[0.0] * len_groupNames for i in range(len(me.vertices))]

    for i, v in enumerate(me.vertices):
        for g in v.groups:
            # possible weights are out of range
            index = g.group
            if index < len_groupNames:
                vWeightList[i][index] = g.weight

    return groupNames, vWeightList

def meshNormalizedWeights(ob, me):
    groupNames, vWeightList = BPyMesh_meshWeight2List(ob, me)

    if not groupNames:
        return [], []

    for i, vWeights in enumerate(vWeightList):
        tot = 0.0
        for w in vWeights:
            tot += w

        if tot:
            for j, w in enumerate(vWeights):
                vWeights[j] = w / tot

    return groupNames, vWeightList

def save_single(operator, scene, filepath="",
        global_matrix=None,
        context_objects=None,
        object_types={'EMPTY', 'ARMATURE', 'MESH'},
        use_mesh_modifiers=True,
        mesh_smooth_type='FACE',
        use_armature_deform_only=False,
        version='BIN',
        use_anim=True,
        use_anim_optimize=True,
        anim_optimize_precision=6,
        use_anim_action_all=False,
        use_metadata=True,
        path_mode='AUTO',
        use_mesh_edges=True,
        use_default_take=True,
        embed_textures=False,
        use_mesh_modifiers_render=True,
        **kwargs
    ):
    
    # Clear cached ObjectWrappers (just in case...).
    ObjectWrapper.cache_clear()    
    
    # Used for mesh and armature rotations
    mtx4_z90 = Matrix.Rotation(math.pi / 2.0, 4, 'Z')
    
    if global_matrix is None:
        global_matrix = Matrix()
        global_scale = 1.0
    else:
        global_scale = global_matrix.median_scale    
    
    fbxSDKExport = FBXExport(5)
    
    if version == 'BIN':
        fbxSDKExport.set_as_ascii(False)
    else:
        fbxSDKExport.set_as_ascii(True)
        
    media_settings = FBXExportSettingsMedia(
        path_mode,
        os.path.dirname(bpy.data.filepath),  # base_src
        os.path.dirname(filepath),  # base_dst
        # Local dir where to put images (medias), using FBX conventions.
        os.path.splitext(os.path.basename(filepath))[0] + ".fbm",  # subdir
        embed_textures,
        set(),  # copy_set
        set(),  # embedded_set
    )
    
    if scene.world:
        data_world = OrderedDict(((scene.world, get_blenderID_key(scene.world)),))
    else:
        data_world = OrderedDict()    
    
    class my_bone_class(object):
        __slots__ = ("blenName",
                     "blenBone",
                     "blenMeshes",
                     "restMatrix",
                     "parent",
                     "blenName",
                     "fbxName",
                     "fbxArm",
                     "__pose_bone",
                     "__anim_poselist")

        def __init__(self, blenBone, fbxArm):

            # This is so 2 armatures dont have naming conflicts since FBX bones use object namespace
            self.fbxName = sane_obname(blenBone)

            self.blenName = blenBone.name
            self.blenBone = blenBone
            self.blenMeshes = {}  # fbxMeshObName : mesh
            self.fbxArm = fbxArm
            self.restMatrix = blenBone.matrix_local

            # not used yet
            #~ self.restMatrixInv = self.restMatrix.inverted()
            #~ self.restMatrixLocal = None # set later, need parent matrix

            self.parent = None

            # not public
            pose = fbxArm.blenObject.pose
            self.__pose_bone = pose.bones[self.blenName]

            # store a list if matrices here, (poseMatrix, head, tail)
            # {frame:posematrix, frame:posematrix, ...}
            self.__anim_poselist = {}

        '''
        def calcRestMatrixLocal(self):
            if self.parent:
                self.restMatrixLocal = self.restMatrix * self.parent.restMatrix.inverted()
            else:
                self.restMatrixLocal = self.restMatrix.copy()
        '''
        def setPoseFrame(self, f):
            # cache pose info here, frame must be set beforehand

            # Didnt end up needing head or tail, if we do - here it is.
            '''
            self.__anim_poselist[f] = (\
                self.__pose_bone.poseMatrix.copy(),\
                self.__pose_bone.head.copy(),\
                self.__pose_bone.tail.copy() )
            '''

            self.__anim_poselist[f] = self.__pose_bone.matrix.copy()

        def getPoseBone(self):
            return self.__pose_bone

        # get pose from frame.
        def getPoseMatrix(self, f):  # ----------------------------------------------
            return self.__anim_poselist[f]
        '''
        def getPoseHead(self, f):
            #return self.__pose_bone.head.copy()
            return self.__anim_poselist[f][1].copy()
        def getPoseTail(self, f):
            #return self.__pose_bone.tail.copy()
            return self.__anim_poselist[f][2].copy()
        '''
        # end

        def getAnimParRelMatrix(self, frame):
            #arm_mat = self.fbxArm.matrixWorld
            #arm_mat = self.fbxArm.parRelMatrix()
            if not self.parent:
                #return mtx4_z90 * (self.getPoseMatrix(frame) * arm_mat) # dont apply arm matrix anymore
                return self.getPoseMatrix(frame) * mtx4_z90
            else:
                #return (mtx4_z90 * ((self.getPoseMatrix(frame) * arm_mat)))  *  (mtx4_z90 * (self.parent.getPoseMatrix(frame) * arm_mat)).inverted()
                return (self.parent.getPoseMatrix(frame) * mtx4_z90).inverted() * ((self.getPoseMatrix(frame)) * mtx4_z90)

        # we need thes because cameras and lights modified rotations
        def getAnimParRelMatrixRot(self, frame):
            return self.getAnimParRelMatrix(frame)

        def flushAnimData(self):
            self.__anim_poselist.clear()
    
    
    class my_object_generic(object):
        __slots__ = ("fbxName",
                     "blenObject",
                     "blenData",
                     "origData",
                     "blenTextures",
                     "blenMaterials",
                     "blenMaterialList",
                     "blenAction",
                     "blenActionList",
                     "fbxGroupNames",
                     "fbxParent",
                     "fbxBoneParent",
                     "fbxBones",
                     "fbxArm",
                     "matrixWorld",
                     "__anim_poselist",
                     )

        # Other settings can be applied for each type - mesh, armature etc.
        def __init__(self, ob, matrixWorld=None):
            self.fbxName = sane_obname(ob)
            self.blenObject = ob
            self.fbxGroupNames = []
            self.fbxParent = None  # set later on IF the parent is in the selection.
            self.fbxArm = None
            if matrixWorld:
                self.matrixWorld = global_matrix * matrixWorld
            else:
                self.matrixWorld = global_matrix * ob.matrix_world

            self.__anim_poselist = {}  # we should only access this

        def parRelMatrix(self):
            if self.fbxParent:
                return self.fbxParent.matrixWorld.inverted() * self.matrixWorld
            else:
                return self.matrixWorld

        def setPoseFrame(self, f, fake=False):
            if fake:
                self.__anim_poselist[f] = self.matrixWorld * global_matrix.inverted()
            else:
                self.__anim_poselist[f] = self.blenObject.matrix_world.copy()

        def getAnimParRelMatrix(self, frame):
            if self.fbxParent:
                #return (self.__anim_poselist[frame] * self.fbxParent.__anim_poselist[frame].inverted() ) * global_matrix
                return (global_matrix * self.fbxParent.__anim_poselist[frame]).inverted() * (global_matrix * self.__anim_poselist[frame])
            else:
                return global_matrix * self.__anim_poselist[frame]

        def getAnimParRelMatrixRot(self, frame):
            obj_type = self.blenObject.type
            if self.fbxParent:
                matrix_rot = ((global_matrix * self.fbxParent.__anim_poselist[frame]).inverted() * (global_matrix * self.__anim_poselist[frame])).to_3x3()
            else:
                matrix_rot = (global_matrix * self.__anim_poselist[frame]).to_3x3()

            return matrix_rot
        
    print('\nFBX SDK export starting... %r' % filepath)
    start_time = time.process_time()    
    
    pose_items = []  # list of (fbxName, matrix) to write pose data for, easier to collect along the way

    # ----------------------------------------------
    def object_tx(ob, loc, matrix, matrix_mod=None):
        """
        Matrix mod is so armature objects can modify their bone matrices
        """
        if isinstance(ob, bpy.types.Bone):

            # we know we have a matrix
            # matrix = mtx4_z90 * (ob.matrix['ARMATURESPACE'] * matrix_mod)
            matrix = ob.matrix_local * mtx4_z90  # dont apply armature matrix anymore

            parent = ob.parent
            if parent:
                #par_matrix = mtx4_z90 * (parent.matrix['ARMATURESPACE'] * matrix_mod)
                par_matrix = parent.matrix_local * mtx4_z90  # dont apply armature matrix anymore
                matrix = par_matrix.inverted() * matrix

            loc, rot, scale = matrix.decompose()
            matrix_rot = rot.to_matrix()

            loc = tuple(loc)
            rot = tuple(rot.to_euler())  # quat -> euler
            scale = tuple(scale)

        else:
            # This is bad because we need the parent relative matrix from the fbx parent (if we have one), dont use anymore
            #if ob and not matrix: matrix = ob.matrix_world * global_matrix
            if ob and not matrix:
                raise Exception("error: this should never happen!")

            matrix_rot = matrix
            #if matrix:
            #    matrix = matrix_scale * matrix

            if matrix:
                loc, rot, scale = matrix.decompose()
                matrix_rot = rot.to_matrix()

                loc = tuple(loc)
                rot = tuple(matrix_rot.to_euler())
                scale = tuple(scale)
            else:
                if not loc:
                    loc = 0.0, 0.0, 0.0
                scale = 1.0, 1.0, 1.0
                rot = 0.0, 0.0, 0.0

        return loc, rot, scale, matrix, matrix_rot
    
    def write_object_tx(ob, loc, matrix, matrix_mod=None):
        """
        We have loc to set the location if non blender objects that have a location

        matrix_mod is only used for bones at the moment
        """
        loc, rot, scale, matrix, matrix_rot = object_tx(ob, loc, matrix, matrix_mod)
        
        return loc, rot, scale, matrix, matrix_rot
    
    def write_object_props(ob=None, loc=None, matrix=None, matrix_mod=None, pose_bone=None):
        
        loc, rot, scale, matrix, matrix_rot = write_object_tx(ob, loc, matrix, matrix_mod)
        
        return loc, rot, scale, matrix, matrix_rot
    
    def write_bone(my_bone):
        #~ poseMatrix = write_object_props(my_bone.blenBone, None, None, my_bone.fbxArm.parRelMatrix())[3]
        loc, rot, scale, matrix, matrix_rot = write_object_props(my_bone.blenBone, pose_bone=my_bone.getPoseBone())  # dont apply bone matrices anymore
        
        fbx_loc = Vector3(loc[0], loc[1], loc[2])
        rot = tuple_rad_to_deg(rot)
        fbx_rot = Vector3(rot[0], rot[1], rot[2])
        fbx_scale = Vector3(scale[0], scale[1], scale[2])
        fbxSDKExport.add_bone(c_char_p(my_bone.fbxName.encode('utf-8')), byref(fbx_loc), byref(fbx_rot), byref(fbx_scale))
        
        global_matrix_bone = (my_bone.fbxArm.matrixWorld * my_bone.restMatrix) * mtx4_z90
        pose_items.append((my_bone.fbxName, global_matrix_bone))        
        
    def write_null(my_null=None, fbxName=None, fbxType="Null", fbxTypeFlags="Null"):
        if not fbxName:
            fbxName = my_null.fbxName
            
        if my_null:
            loc, rot, scale, poseMatrix, matrix_rot = write_object_props(my_null.blenObject, None, my_null.parRelMatrix())
        else:
            loc, rot, scale, poseMatrix, matrix_rot = write_object_props()
            
        fbx_loc = Vector3(loc[0], loc[1], loc[2])
        rot = tuple_rad_to_deg(rot)
        fbx_rot = Vector3(rot[0], rot[1], rot[2])
        fbx_scale = Vector3(scale[0], scale[1], scale[2])
        fbxSDKExport.add_bone(c_char_p(fbxName.encode('utf-8')), byref(fbx_loc), byref(fbx_rot), byref(fbx_scale))
        
        pose_items.append((fbxName, poseMatrix))

    def write_material(matname, mat):
        if mat:
            mat_shadeless = mat.use_shadeless
            if mat_shadeless:
                mat_shader = 'Lambert'
            else:
                if mat.diffuse_shader == 'LAMBERT':
                    mat_shader = 'Lambert'
                else:
                    mat_shader = 'Phong'
                    
        else:
            mat_shader = 'Phong'            
        
        
        
    def write_sub_deformer_skin(my_mesh, my_bone, weights):
        if my_mesh.fbxBoneParent:
            if my_mesh.fbxBoneParent == my_bone:
                # TODO - this is a bit lazy, we could have a simple write loop
                # for this case because all weights are 1.0 but for now this is ok
                # Parent Bones arent used all that much anyway.
                vgroup_data = [(j, 1.0) for j in range(len(my_mesh.blenData.vertices))]
            else:
                # This bone is not a parent of this mesh object, no weights
                vgroup_data = []

        else:
            # Normal weight painted mesh
            if my_bone.blenName in weights[0]:
                # Before we used normalized weight list
                group_index = weights[0].index(my_bone.blenName)
                vgroup_data = [(j, weight[group_index]) for j, weight in enumerate(weights[1]) if weight[group_index]]
            else:
                vgroup_data = []
                
        for vg in vgroup_data:
            fbxSDKExport.add_sub_deformer_index(c_char_p(my_mesh.fbxName.encode('utf-8')), c_char_p(my_bone.fbxName.encode('utf-8')), vg[0])
            fbxSDKExport.add_sub_deformer_weight(c_char_p(my_mesh.fbxName.encode('utf-8')), c_char_p(my_bone.fbxName.encode('utf-8')), vg[1])
            
        global_bone_matrix = (my_bone.fbxArm.matrixWorld * my_bone.restMatrix) * mtx4_z90
        global_mesh_matrix = my_mesh.matrixWorld
        transform_matrix = (global_bone_matrix.inverted() * global_mesh_matrix)
        
        global_bone_matrix_transp = global_bone_matrix.transposed()
        transform_matrix_transp = global_mesh_matrix.transposed()
        
        fbx_transform_matrix = Mat4x4(transform_matrix_transp[0][0], transform_matrix_transp[0][1], transform_matrix_transp[0][2], transform_matrix_transp[0][3], 
                                      transform_matrix_transp[1][0], transform_matrix_transp[1][1], transform_matrix_transp[1][2], transform_matrix_transp[1][3], 
                                      transform_matrix_transp[2][0], transform_matrix_transp[2][1], transform_matrix_transp[2][2], transform_matrix_transp[2][3], 
                                      transform_matrix_transp[3][0], transform_matrix_transp[3][1], transform_matrix_transp[3][2], transform_matrix_transp[3][3])
        #fbx_quat = Vector4(transform_quat[1], transform_quat[2], transform_quat[3], transform_quat[0])
        fbx_quat = Vector4(0, 0, 0, 0)
        fbxSDKExport.set_sub_deformer_transform(c_char_p(my_mesh.fbxName.encode('utf-8')), c_char_p(my_bone.fbxName.encode('utf-8')), 
                                                byref(fbx_transform_matrix), byref(fbx_quat))
        
        fbx_global_bone_matrix = Mat4x4(global_bone_matrix_transp[0][0], global_bone_matrix_transp[0][1], global_bone_matrix_transp[0][2], global_bone_matrix_transp[0][3], 
                                      global_bone_matrix_transp[1][0], global_bone_matrix_transp[1][1], global_bone_matrix_transp[1][2], global_bone_matrix_transp[1][3], 
                                      global_bone_matrix_transp[2][0], global_bone_matrix_transp[2][1], global_bone_matrix_transp[2][2], global_bone_matrix_transp[2][3], 
                                      global_bone_matrix_transp[3][0], global_bone_matrix_transp[3][1], global_bone_matrix_transp[3][2], global_bone_matrix_transp[3][3])
        fbxSDKExport.set_sub_deformer_transform_link(c_char_p(my_mesh.fbxName.encode('utf-8')), c_char_p(my_bone.fbxName.encode('utf-8')), 
                                                     byref(fbx_global_bone_matrix))
    
    def write_mesh(my_mesh):
        me = my_mesh.blenData
        
        do_materials = bool([m for m in my_mesh.blenMaterials if m is not None])
        do_textures = bool([t for t in my_mesh.blenTextures if t is not None])
        do_uvs = bool(me.uv_layers)
        
        loc, rot, scale, matrix, matrix_rot = write_object_tx(my_mesh.blenObject, None, my_mesh.parRelMatrix())
        rot = tuple_rad_to_deg(rot)       
        
        fbx_loc = Vector3(loc[0], loc[1], loc[2])
        fbx_rot = Vector3(rot[0], rot[1], rot[2])
        fbx_scale = Vector3(scale[0], scale[1], scale[2])
        fbxSDKExport.set_mesh_property(c_char_p(my_mesh.fbxName.encode('utf-8')), byref(fbx_loc), byref(fbx_rot), byref(fbx_scale))
        
        # Calculate the global transform for the mesh in the bind pose the same way we do
        # in write_sub_deformer_skin
        globalMeshBindPose = my_mesh.matrixWorld * mtx4_z90
        pose_items.append((my_mesh.fbxName, globalMeshBindPose))
        
        _nchunk = 3
        t_co = [None] * len(me.vertices) * 3
        me.vertices.foreach_get("co", t_co)
        for vertex in grouper_exact(t_co, _nchunk):
            fbxSDKExport.add_vertex(vertex[0], vertex[1], vertex[2])
        del t_co
        
        t_vi = [None] * len(me.loops)
        me.loops.foreach_get("vertex_index", t_vi)
        for ix in t_vi:
            fbxSDKExport.add_index(ix)
        
        t_ls = [None] * len(me.polygons)
        me.polygons.foreach_get("loop_start", t_ls)
        if t_ls != sorted(t_ls):
            print("Error: polygons and loops orders do not match!")
                    
        for ls in t_ls:
            fbxSDKExport.add_loop_start(ls)
            
        del t_vi
        del t_ls            
        
        if use_mesh_edges:
            t_vi = [None] * len(me.edges) * 2
            me.edges.foreach_get("vertices", t_vi)

            # write loose edges as faces.
            t_el = [None] * len(me.edges)
            me.edges.foreach_get("is_loose", t_el)
            num_lose = sum(t_el)
            '''
            if num_lose != 0:
                it_el = ((vi ^ -1) if (idx % 2) else vi for idx, vi in enumerate(t_vi) if t_el[idx // 2])
                if (len(me.loops)):
                    fw(prep)
                fw(prep.join(','.join('%i' % vi for vi in chunk) for chunk in grouper_exact(it_el, _nchunk)))
            '''
            #fw('\n\t\tEdges: ')
            #fw(',\n\t\t       '.join(','.join('%i' % vi for vi in chunk) for chunk in grouper_exact(t_vi, _nchunk)))
            _npiece = 2
            for edge in grouper_exact(t_vi, _npiece):
                fbxSDKExport.add_mesh_edge(c_char_p(my_mesh.fbxName.encode('utf-8')), edge[0], edge[1])
                    
            del t_vi
            del t_el            
            
        t_vn = [None] * len(me.loops) * 3
        me.calc_normals_split()
        me.loops.foreach_get("normal", t_vn)
        for vnormal in grouper_exact(t_vn, _nchunk):
            fbxSDKExport.add_normal(vnormal[0], vnormal[1], vnormal[2])
            
        del t_vn
        me.free_normals_split()
        
        if mesh_smooth_type == 'FACE':
            # Write Face Smoothing
            fbxSDKExport.set_smoothing_mode(0)
            t_ps = [None] * len(me.polygons)
            me.polygons.foreach_get("use_smooth", t_ps)
            for ps in t_ps:
                fbxSDKExport.add_smoothing(ps)
            del t_ps
        elif mesh_smooth_type == 'EDGE':
            # Write Edge Smoothing
            fbxSDKExport.set_smoothing_mode(1)
            t_es = [None] * len(me.edges)
            me.edges.foreach_get("use_edge_sharp", t_es)
            for es in t_es:
                fbxSDKExport.add_smoothing(es)
            del t_es
        elif mesh_smooth_type == 'OFF':
            fbxSDKExport.set_smoothing_mode(-1)
        else:
            raise Exception("invalid mesh_smooth_type: %r" % mesh_smooth_type)
        
        uvlayers = []
        uvtextures = []
        if do_uvs:
            uvlayers = me.uv_layers
            uvtextures = me.uv_textures
            t_uv = [None] * len(me.loops) * 2
            uv2idx = None
            tex2idx = None
            _nchunk_idx = 64  # Number of UV indices per line
            
            if do_textures:
                is_tex_unique = len(my_mesh.blenTextures) == 1
                tex2idx = {None: -1}
                tex2idx.update({tex: i for i, tex in enumerate(my_mesh.blenTextures)})       
                
            for uvindex, (uvlayer, uvtexture) in enumerate(zip(uvlayers, uvtextures)):
                uvlayer.data.foreach_get("uv", t_uv)
                uvco = tuple(zip(*[iter(t_uv)] * 2))
                fbxSDKExport.create_uv_info(uvindex, c_char_p(uvlayer.name.encode('utf-8')))
                uv2idx = tuple(set(uvco))
                for uv in uv2idx:
                    fbxSDKExport.add_uv(uvindex, uv[0], uv[1])
                uv2idx = {uv: idx for idx, uv in enumerate(uv2idx)}
                for chunk in grouper_exact(uvco, _nchunk_idx):
                    for uv in chunk:
                        fbxSDKExport.add_uv_index(uvindex, uv2idx[uv])    
            
            del t_uv
            
        if do_materials:
            is_mat_unique = len(my_mesh.blenMaterials) == 1
            if is_mat_unique:
                pass
            else:
                _nchunk = 64  # Number of material indices per line
                # Build a material mapping for this
                mat2idx = {mt: i for i, mt in enumerate(my_mesh.blenMaterials)}  # (local-mat, tex) -> global index.
                mats = my_mesh.blenMaterialList
                if me.uv_textures.active and do_uvs:
                    poly_tex = me.uv_textures.active.data
                else:
                    poly_tex = [None] * len(me.polygons)
                _it_mat = (mats[p.material_index] for p in me.polygons)
                _it_tex = (pt.image if pt else None for pt in poly_tex)  # WARNING - MULTI UV LAYER IMAGES NOT SUPPORTED
                t_mti = (mat2idx[m, t] for m, t in zip(_it_mat, _it_tex))
                
                for chunk in grouper_exact(t_mti, _nchunk):
                    for i in chunk:
                        pass
                        #fbxSDKExport.add_mat_index(i)
            
        # add meshes here to clear because they are not used anywhere.
    meshes_to_clear = []
    
    ob_meshes = []
    
    ob_bones = []
    ob_arms = []
    
    # List of types that have blender objects (not bones)
    ob_all_typegroups = [ob_meshes, ob_arms]    
    
    groups = []  # blender groups, only add ones that have objects in the selections
    materials = set()  # (mat, image) items
    textures = set()
    
    if 'ARMATURE' in object_types:
        # This is needed so applying modifiers dosnt apply the armature deformation, its also needed
        # ...so mesh objects return their rest worldspace matrix when bone-parents are exported as weighted meshes.
        # set every armature to its rest, backup the original values so we done mess up the scene
        ob_arms_orig_rest = [arm.pose_position for arm in bpy.data.armatures]

        for arm in bpy.data.armatures:
            arm.pose_position = 'REST'

        if ob_arms_orig_rest:
            for ob_base in bpy.data.objects:
                if ob_base.type == 'ARMATURE':
                    ob_base.update_tag()

            # This causes the makeDisplayList command to effect the mesh
            scene.frame_set(scene.frame_current)    
        
    for ob_base in context_objects:

        # ignore dupli children
        if ob_base.parent and ob_base.parent.dupli_type in {'VERTS', 'FACES'}:
            continue

        obs = [(ob_base, ob_base.matrix_world.copy())]
        if ob_base.dupli_type != 'NONE':
            ob_base.dupli_list_create(scene)
            obs = [(dob.object, dob.matrix.copy()) for dob in ob_base.dupli_list]

        for ob, mtx in obs:
            tmp_ob_type = ob.type
            
            if tmp_ob_type == 'ARMATURE':
                if 'ARMATURE' in object_types:
                    # TODO - armatures dont work in dupligroups!
                    if ob not in ob_arms:
                        ob_arms.append(ob)
                    # ob_arms.append(ob) # replace later. was "ob_arms.append(sane_obname(ob), ob)"
            elif 'MESH' in object_types:
                origData = True
                if tmp_ob_type != 'MESH':
                    try:
                        me = ob.to_mesh(scene, True, 'PREVIEW')
                    except:
                        me = None

                    if me:
                        meshes_to_clear.append(me)
                        mats = me.materials
                        origData = False
                else:
                    # Mesh Type!
                    if use_mesh_modifiers:
                        me = ob.to_mesh(scene, True, 'PREVIEW')

                        # print ob, me, me.getVertGroupNames()
                        meshes_to_clear.append(me)
                        origData = False
                        mats = me.materials
                    else:
                        me = ob.data
                        me.update()
                        mats = me.materials

                if me:
#                     # This WILL modify meshes in blender if use_mesh_modifiers is disabled.
#                     # so strictly this is bad. but only in rare cases would it have negative results
#                     # say with dupliverts the objects would rotate a bit differently
#                     if EXP_MESH_HQ_NORMALS:
#                         BPyMesh.meshCalcNormals(me) # high quality normals nice for realtime engines.
                    

                    if not mats:
                        mats = [None]

                    texture_set_local = set()
                    material_set_local = set()
                    if me.uv_textures:
                        for uvlayer in me.uv_textures:
                            for p, p_uv in zip(me.polygons, uvlayer.data):
                                tex = p_uv.image
                                texture_set_local.add(tex)
                                mat = mats[p.material_index]

                                # Should not be needed anymore.
                                #try:
                                    #mat = mats[p.material_index]
                                #except:
                                    #mat = None

                                material_set_local.add((mat, tex))

                    else:
                        for mat in mats:
                            # 2.44 use mat.lib too for uniqueness
                            material_set_local.add((mat, None))

                    textures |= texture_set_local
                    materials |= material_set_local

                    if 'ARMATURE' in object_types:
                        armob = ob.find_armature()
                        blenParentBoneName = None

                        # parent bone - special case
                        if (not armob) and ob.parent and ob.parent.type == 'ARMATURE' and \
                                ob.parent_type == 'BONE':
                            armob = ob.parent
                            blenParentBoneName = ob.parent_bone

                        if armob and armob not in ob_arms:
                            ob_arms.append(armob)

                        # Warning for scaled, mesh objects with armatures
                        if abs(ob.scale[0] - 1.0) > 0.05 or abs(ob.scale[1] - 1.0) > 0.05 or abs(ob.scale[1] - 1.0) > 0.05:
                            operator.report(
                                    {'WARNING'},
                                    "Object '%s' has a scale of (%.3f, %.3f, %.3f), "
                                    "Armature deformation will not work as expected "
                                    "(apply Scale to fix)" % (ob.name, *ob.scale))

                    else:
                        blenParentBoneName = armob = None

                    my_mesh = my_object_generic(ob, mtx)
                    my_mesh.blenData = me
                    my_mesh.origData = origData
                    my_mesh.blenMaterials = list(material_set_local)
                    my_mesh.blenMaterialList = mats
                    my_mesh.blenTextures = list(texture_set_local)

                    # sort the name so we get predictable output, some items may be NULL
                    my_mesh.blenMaterials.sort(key=lambda m: (getattr(m[0], "name", ""), getattr(m[1], "name", "")))
                    my_mesh.blenTextures.sort(key=lambda m: getattr(m, "name", ""))

                    # if only 1 null texture then empty the list
                    if len(my_mesh.blenTextures) == 1 and my_mesh.blenTextures[0] is None:
                        my_mesh.blenTextures = []

                    my_mesh.fbxArm = armob  # replace with my_object_generic armature instance later
                    my_mesh.fbxBoneParent = blenParentBoneName  # replace with my_bone instance later

                    ob_meshes.append(my_mesh)

        # not forgetting to free dupli_list
        if ob_base.dupli_list:
            ob_base.dupli_list_clear()
            
    if 'ARMATURE' in object_types:
        # now we have the meshes, restore the rest arm position
        for i, arm in enumerate(bpy.data.armatures):
            arm.pose_position = ob_arms_orig_rest[i]

        if ob_arms_orig_rest:
            for ob_base in bpy.data.objects:
                if ob_base.type == 'ARMATURE':
                    ob_base.update_tag()
            # This causes the makeDisplayList command to effect the mesh
            scene.frame_set(scene.frame_current)

    del tmp_ob_type

    # now we have collected all armatures, add bones
    for i, ob in enumerate(ob_arms):

        ob_arms[i] = my_arm = my_object_generic(ob)

        my_arm.fbxBones = []
        my_arm.blenData = ob.data
        if ob.animation_data:
            my_arm.blenAction = ob.animation_data.action
        else:
            my_arm.blenAction = None
        my_arm.blenActionList = []

        # fbxName, blenderObject, my_bones, blenderActions
        #ob_arms[i] = fbxArmObName, ob, arm_my_bones, (ob.action, [])

        if use_armature_deform_only:
            # tag non deforming bones that have no deforming children
            deform_map = dict.fromkeys(my_arm.blenData.bones, False)
            for bone in my_arm.blenData.bones:
                if bone.use_deform:
                    deform_map[bone] = True
                    # tag all parents, even ones that are not deform since their child _is_
                    for parent in bone.parent_recursive:
                        deform_map[parent] = True

        for bone in my_arm.blenData.bones:

            if use_armature_deform_only:
                # if this bone doesnt deform, and none of its children deform, skip it!
                if not deform_map[bone]:
                    continue

            my_bone = my_bone_class(bone, my_arm)
            my_arm.fbxBones.append(my_bone)
            ob_bones.append(my_bone)

        if use_armature_deform_only:
            del deform_map

    # add the meshes to the bones and replace the meshes armature with own armature class
    #for obname, ob, mtx, me, mats, arm, armname in ob_meshes:
    for my_mesh in ob_meshes:
        # Replace
        # ...this could be sped up with dictionary mapping but its unlikely for
        # it ever to be a bottleneck - (would need 100+ meshes using armatures)
        if my_mesh.fbxArm:
            for my_arm in ob_arms:
                if my_arm.blenObject == my_mesh.fbxArm:
                    my_mesh.fbxArm = my_arm
                    break

        for my_bone in ob_bones:

            # The mesh uses this bones armature!
            if my_bone.fbxArm == my_mesh.fbxArm:
                if my_bone.blenBone.use_deform:
                    my_bone.blenMeshes[my_mesh.fbxName] = me

                # parent bone: replace bone names with our class instances
                # my_mesh.fbxBoneParent is None or a blender bone name initialy, replacing if the names match.
                if my_mesh.fbxBoneParent == my_bone.blenName:
                    my_mesh.fbxBoneParent = my_bone

    bone_deformer_count = 0  # count how many bones deform a mesh
    my_bone_blenParent = None
    for my_bone in ob_bones:
        my_bone_blenParent = my_bone.blenBone.parent
        if my_bone_blenParent:
            for my_bone_parent in ob_bones:
                # Note 2.45rc2 you can compare bones normally
                if my_bone_blenParent.name == my_bone_parent.blenName and my_bone.fbxArm == my_bone_parent.fbxArm:
                    my_bone.parent = my_bone_parent
                    break

        # Not used at the moment
        # my_bone.calcRestMatrixLocal()
        bone_deformer_count += len(my_bone.blenMeshes)

    del my_bone_blenParent
    
    # Build blenObject -> fbxObject mapping
    # this is needed for groups as well as fbxParenting
    bpy.data.objects.tag(False)

    # using a list of object names for tagging (Arystan)

    tmp_obmapping = {}
    for ob_generic in ob_all_typegroups:
        for ob_base in ob_generic:
            ob_base.blenObject.tag = True
            tmp_obmapping[ob_base.blenObject] = ob_base

    # Build Groups from objects we export
    for blenGroup in bpy.data.groups:
        fbxGroupName = None
        for ob in blenGroup.objects:
            if ob.tag:
                if fbxGroupName is None:
                    fbxGroupName = sane_groupname(blenGroup)
                    groups.append((fbxGroupName, blenGroup))

                tmp_obmapping[ob].fbxGroupNames.append(fbxGroupName)  # also adds to the objects fbxGroupNames

    groups.sort()  # not really needed

    # Assign parents using this mapping
    for ob_generic in ob_all_typegroups:
        for my_ob in ob_generic:
            parent = my_ob.blenObject.parent
            if parent and parent.tag:  # does it exist and is it in the mapping
                my_ob.fbxParent = tmp_obmapping[parent]

    del tmp_obmapping
    # Finished finding groups we use
    
    def fbx_data_mesh_elements(root, me_obj, mesh_mat_indices, data_meshes):
        me_key, me, _free = data_meshes[me_obj]
        
        me_fbxmats_idx = mesh_mat_indices.get(me)
        if me_fbxmats_idx is not None:
            me_blmats = me.materials
            if me_fbxmats_idx and me_blmats:
                nbr_mats = len(me_fbxmats_idx)
                if nbr_mats > 1:
                    t_pm = array.array(data_types.ARRAY_INT32, (0,)) * len(me.polygons)
                    me.polygons.foreach_get("material_index", t_pm)
    
                    # We have to validate mat indices, and map them to FBX indices.
                    # Note a mat might not be in me_fbxmats_idx (e.g. node mats are ignored).
                    blmats_to_fbxmats_idxs = [me_fbxmats_idx[m] for m in me_blmats if m in me_fbxmats_idx]
                    mat_idx_limit = len(blmats_to_fbxmats_idxs)
                    def_mat = blmats_to_fbxmats_idxs[0]
                    _gen = (blmats_to_fbxmats_idxs[m] if m < mat_idx_limit else def_mat for m in t_pm)
                    t_pm = array.array(data_types.ARRAY_INT32, _gen)
                    
                    for pm in t_pm:
                        fbxSDKExport.add_mat_index(c_char_p(ob_obj.name.encode('utf-8')), pm)
                    del t_pm
        
        
    
    def fbx_data_material_elements(root, mat):
        ambient_color = (0.0, 0.0, 0.0)
        if data_world:
            ambient_color = next(iter(data_world.keys())).ambient_color
        skip_mat = check_skip_material(mat)
        node_mat = mat.use_nodes            
        mat_type = b"Phong"
        # Approximation...
        if not skip_mat and not node_mat and mat.specular_shader not in {'COOKTORR', 'PHONG', 'BLINN'}:
            mat_type = b"Lambert"

        fbx_diffuse = Vector3(mat.diffuse_color[0], mat.diffuse_color[1], mat.diffuse_color[2])
        fbx_ambient = Vector3(ambient_color[0], ambient_color[1], ambient_color[2])
        fbx_emissive = fbx_diffuse
            
        fbxSDKExport.add_material(c_char_p(mat.name.encode('utf-8')), mat_type, byref(fbx_diffuse), byref(fbx_ambient), byref(fbx_emissive))
    
    def check_skip_material(mat):
        """Simple helper to check whether we actually support exporting that material or not"""
        return mat.type not in {'SURFACE'}    
    
    def _gen_vid_path(img):
        msetts = media_settings
        fname_rel = bpy_extras.io_utils.path_reference(img.filepath, msetts.base_src, msetts.base_dst, msetts.path_mode,
                                                       msetts.subdir, msetts.copy_set, img.library)
        fname_abs = os.path.normpath(os.path.abspath(os.path.join(msetts.base_dst, fname_rel)))
        return fname_abs, fname_rel    
    
    def fbx_data_texture_file_elements(root, tex, data_textures):
        tex_key, _mats = data_textures[tex]
        img = tex.texture.image
        fname_abs, fname_rel = _gen_vid_path(img)
        
        alpha_source = 0  # None
        if img.use_alpha:
            if tex.texture.use_calculate_alpha:
                alpha_source = 1  # RGBIntensity as alpha.
            else:
                alpha_source = 2  # Black, i.e. alpha channel.
                
        # BlendMode not useful for now, only affects layered textures afaics.
        mapping = 0  # UV.
        uvset = None
        if tex.texture_coords in {'ORCO'}:  # XXX Others?
            if tex.mapping in {'FLAT'}:
                mapping = 1  # Planar
            elif tex.mapping in {'CUBE'}:
                mapping = 4  # Box
            elif tex.mapping in {'TUBE'}:
                mapping = 3  # Cylindrical
            elif tex.mapping in {'SPHERE'}:
                mapping = 2  # Spherical
        elif tex.texture_coords in {'UV'}:
            mapping = 0  # UV
            # Yuck, UVs are linked by mere names it seems... :/
            uvset = tex.uv_layer
        wrap_mode = 1  # Clamp
        if tex.texture.extension in {'REPEAT'}:
            wrap_mode = 0  # Repeat            
                
        fbx_translation = Vector3(tex.offset[0], tex.offset[1], tex.offset[2])
        fbx_scaling = Vector3(tex.scale[0], tex.scale[1], tex.scale[2])
        
        fbxSDKExport.add_texture(c_char_p(tex.name.encode('utf-8')), c_char_p(fname_abs.encode('utf-8')), c_char_p(fname_rel.encode('utf-8')), 
                                 alpha_source, img.alpha_mode in {'STRAIGHT'}, mapping, c_char_p(uvset.encode('utf-8')) if uvset is not None else "", 
                                 wrap_mode, wrap_mode, byref(fbx_translation), byref(fbx_scaling), True, tex.texture.use_mipmap)        
        
    def fbx_mat_properties_from_texture(tex):
        """
        Returns a set of FBX metarial properties that are affected by the given texture.
        Quite obviously, this is a fuzzy and far-from-perfect mapping! Amounts of influence are completely lost, e.g.
        Note tex is actually expected to be a texture slot.
        """
        # Mapping Blender -> FBX (blend_use_name, blend_fact_name, fbx_name).
        blend_to_fbx = (
            # Lambert & Phong...
            ("diffuse", "diffuse", b"DiffuseFactor"),
            ("color_diffuse", "diffuse_color", b"DiffuseColor"),
            ("alpha", "alpha", b"TransparencyFactor"),
            ("diffuse", "diffuse", b"TransparentColor"),  # Uses diffuse color in Blender!
            ("emit", "emit", b"EmissiveFactor"),
            ("diffuse", "diffuse", b"EmissiveColor"),  # Uses diffuse color in Blender!
            ("ambient", "ambient", b"AmbientFactor"),
            # ("", "", b"AmbientColor"),  # World stuff in Blender, for now ignore...
            ("normal", "normal", b"NormalMap"),
            # Note: unsure about those... :/
            # ("", "", b"Bump"),
            # ("", "", b"BumpFactor"),
            # ("", "", b"DisplacementColor"),
            # ("", "", b"DisplacementFactor"),
            # Phong only.
            ("specular", "specular", b"SpecularFactor"),
            ("color_spec", "specular_color", b"SpecularColor"),
            # See Material template about those two!
            ("hardness", "hardness", b"Shininess"),
            ("hardness", "hardness", b"ShininessExponent"),
            ("mirror", "mirror", b"ReflectionColor"),
            ("raymir", "raymir", b"ReflectionFactor"),
        )
    
        tex_fbx_props = set()
        for use_map_name, name_factor, fbx_prop_name in blend_to_fbx:
            # Always export enabled textures, even if they have a null influence...
            if getattr(tex, "use_map_" + use_map_name):
                tex_fbx_props.add(fbx_prop_name)
    
        return tex_fbx_props        
    
    objects = OrderedDict()  # Because we do not have any ordered set...
    objtypes = object_types
    for ob in context_objects:
        if ob.type not in objtypes:
            continue
        ob_obj = ObjectWrapper(ob)
        objects[ob_obj] = None
        
    data_meshes = OrderedDict()
    for ob_obj in objects:
        if ob_obj.type not in BLENDER_OBJECT_TYPES_MESHLIKE:
            continue
        ob = ob_obj.bdata
        use_org_data = True
        org_ob_obj = None

        # Do not want to systematically recreate a new mesh for dupliobject instances, kind of break purpose of those.
        if ob_obj.is_dupli:
            org_ob_obj = ObjectWrapper(ob)  # We get the "real" object wrapper from that dupli instance.
            if org_ob_obj in data_meshes:
                data_meshes[ob_obj] = data_meshes[org_ob_obj]
                continue

        is_ob_material = any(ms.link == 'OBJECT' for ms in ob.material_slots)

        if use_mesh_modifiers or ob.type in BLENDER_OTHER_OBJECT_TYPES or is_ob_material:
            # We cannot use default mesh in that case, or material would not be the right ones...
            use_org_data = not (is_ob_material or ob.type in BLENDER_OTHER_OBJECT_TYPES)
            tmp_mods = []
            if use_org_data and ob.type == 'MESH':
                # No need to create a new mesh in this case, if no modifier is active!
                for mod in ob.modifiers:
                    # For meshes, when armature export is enabled, disable Armature modifiers here!
                    if mod.type == 'ARMATURE' and 'ARMATURE' in object_types:
                        tmp_mods.append((mod, mod.show_render))
                        mod.show_render = False
                    if mod.show_render:
                        use_org_data = False
            if not use_org_data:
                tmp_me = ob.to_mesh(
                    scene,
                    apply_modifiers=use_mesh_modifiers,
                    settings='RENDER' if use_mesh_modifiers_render else 'PREVIEW',
                )
                data_meshes[ob_obj] = (get_blenderID_key(tmp_me), tmp_me, True)
            # Re-enable temporary disabled modifiers.
            for mod, show_render in tmp_mods:
                mod.show_render = show_render
        if use_org_data:
            data_meshes[ob_obj] = (get_blenderID_key(ob.data), ob.data, False)

        # In case "real" source object of that dupli did not yet still existed in data_meshes, create it now!
        if org_ob_obj is not None:
            data_meshes[org_ob_obj] = data_meshes[ob_obj]        
        
    data_materials = OrderedDict()
    for ob_obj in objects:
        # If obj is not a valid object for materials, wrapper will just return an empty tuple...
        for mat_s in ob_obj.material_slots:
            mat = mat_s.material
            if mat is None:
                continue  # Empty slots!
            # Note theoretically, FBX supports any kind of materials, even GLSL shaders etc.
            # However, I doubt anything else than Lambert/Phong is really portable!
            # We support any kind of 'surface' shader though, better to have some kind of default Lambert than nothing.
            # Note we want to keep a 'dummy' empty mat even when we can't really support it, see T41396.
            mat_data = data_materials.get(mat)
            if mat_data is not None:
                mat_data[1].append(ob_obj)
            else:
                data_materials[mat] = (get_blenderID_key(mat), [ob_obj])        
        
    # Note FBX textures also hold their mapping info.
    # TODO: Support layers?
    data_textures = OrderedDict()
    # FbxVideo also used to store static images...
    data_videos = OrderedDict()
    # For now, do not use world textures, don't think they can be linked to anything FBX wise...
    for mat in data_materials.keys():
        if check_skip_material(mat):
            continue
        for tex, use_tex in zip(mat.texture_slots, mat.use_textures):
            if tex is None or tex.texture is None or not use_tex:
                continue
            # For now, only consider image textures.
            # Note FBX does has support for procedural, but this is not portable at all (opaque blob),
            # so not useful for us.
            # TODO I think ENVIRONMENT_MAP should be usable in FBX as well, but for now let it aside.
            # if tex.texture.type not in {'IMAGE', 'ENVIRONMENT_MAP'}:
            if tex.texture.type not in {'IMAGE'}:
                continue
            img = tex.texture.image
            if img is None:
                continue
            # Find out whether we can actually use this texture for this material, in FBX context.
            tex_fbx_props = fbx_mat_properties_from_texture(tex)
            if not tex_fbx_props:
                continue
            tex_data = data_textures.get(tex)
            if tex_data is not None:
                tex_data[1][mat] = tex_fbx_props
            else:
                data_textures[tex] = (get_blenderID_key(tex), OrderedDict(((mat, tex_fbx_props),)))
            vid_data = data_videos.get(img)
            if vid_data is not None:
                vid_data[1].append(tex)
            else:
                data_videos[img] = (get_blenderID_key(img), [tex])
                
    # Materials
    mesh_mat_indices = OrderedDict()
    _objs_indices = {}
    for mat, (mat_key, ob_objs) in data_materials.items():
        for ob_obj in ob_objs:
            #connections.append((b"OO", get_fbx_uuid_from_key(mat_key), ob_obj.fbx_uuid, None))
            # Get index of this mat for this object (or dupliobject).
            # Mat indices for mesh faces are determined by their order in 'mat to ob' connections.
            # Only mats for meshes currently...
            # Note in case of dupliobjects a same me/mat idx will be generated several times...
            # Should not be an issue in practice, and it's needed in case we export duplis but not the original!
            if ob_obj.type not in BLENDER_OBJECT_TYPES_MESHLIKE:
                continue
            _mesh_key, me, _free = data_meshes[ob_obj]
            idx = _objs_indices[ob_obj] = _objs_indices.get(ob_obj, -1) + 1
            mesh_mat_indices.setdefault(me, OrderedDict())[mat] = idx
    del _objs_indices


    for me_obj in data_meshes:
        fbx_data_mesh_elements(objects, me_obj, mesh_mat_indices, data_meshes)                
                
    for mat in data_materials:
        fbx_data_material_elements(objects, mat)                
                
    for tex in data_textures:
        fbx_data_texture_file_elements(objects, tex, data_textures)
        
    # Textures
    for tex, (tex_key, mats) in data_textures.items():
        for mat, fbx_mat_props in mats.items():
            mat_key, _ob_objs = data_materials[mat]
            for fbx_prop in fbx_mat_props:
                # texture -> material properties
                #connections.append((b"OP", get_fbx_uuid_from_key(tex_key), get_fbx_uuid_from_key(mat_key), fbx_prop))
                fbxSDKExport.set_texture_mat_prop(c_char_p(tex.name.encode('utf-8')), c_char_p(mat.name.encode('utf-8')), c_char_p(fbx_prop))

                
    del context_objects, objects

    # == WRITE OBJECTS TO THE FILE ==
    # == From now on we are building the FBX file from the information collected above (JCB)    
    
    materials = [(sane_matname(mat_tex_pair), mat_tex_pair) for mat_tex_pair in materials]
    textures = [(sane_texname(tex), tex) for tex in textures if tex]
    materials.sort(key=lambda m: m[0])  # sort by name
    textures.sort(key=lambda m: m[0])    
        
    # sanity checks
    try:
        assert(not (ob_meshes and ('MESH' not in object_types)))
        assert(not (materials and ('MESH' not in object_types)))
        assert(not (textures and ('MESH' not in object_types)))
        
    except AssertionError:
        import traceback
        traceback.print_exc()
        
    for my_arm in ob_arms:
        write_null(my_arm, fbxType="Limb", fbxTypeFlags="Skeleton")
        
    for my_mesh in ob_meshes:
        write_mesh(my_mesh)

    #for bonename, bone, obname, me, armob in ob_bones:
    for my_bone in ob_bones:
        write_bone(my_bone)        
        
    #for matname, (mat, tex) in materials:
        #write_material(matname, mat)  # We only need to have a material per image pair, but no need to write any image info into the material (dumb fbx standard)        
    
    for my_mesh in ob_meshes:
        if my_mesh.fbxArm:
            if my_mesh.fbxBoneParent:
                weights = None
            else:
                weights = meshNormalizedWeights(my_mesh.blenObject, my_mesh.blenData)
                
                
            for my_bone in ob_bones:
                if me in iter(my_bone.blenMeshes.values()):
                    write_sub_deformer_skin(my_mesh, my_bone, weights)
                    
    for fbxName, _matrix in pose_items:
        matrix = _matrix if _matrix else Matrix()
        matrix_transp = matrix.transposed()
        fbx_matrix_transp = Mat4x4(matrix_transp[0][0], matrix_transp[0][1], matrix_transp[0][2], matrix_transp[0][3], 
                                      matrix_transp[1][0], matrix_transp[1][1], matrix_transp[1][2], matrix_transp[1][3], 
                                      matrix_transp[2][0], matrix_transp[2][1], matrix_transp[2][2], matrix_transp[2][3], 
                                      matrix_transp[3][0], matrix_transp[3][1], matrix_transp[3][2], matrix_transp[3][3])
        fbxSDKExport.add_pose_node(c_char_p(fbxName.encode('utf-8')), byref(fbx_matrix_transp))
    
    for my_bone in ob_bones:
        # Always parent to armature now
        if my_bone.parent:
            fbxSDKExport.add_bone_child(c_char_p(my_bone.fbxName.encode('utf-8')), c_char_p(my_bone.parent.fbxName.encode('utf-8')))
        else:
            # the armature object is written as an empty and all root level bones connect to it
            fbxSDKExport.add_bone_child(c_char_p(my_bone.fbxName.encode('utf-8')), c_char_p(my_bone.fbxArm.fbxName.encode('utf-8')))
            
    # Needed for scene footer as well as animation
    render = scene.render
    
    fps = float(render.fps)
    start = scene.frame_start
    end = scene.frame_end
    if end < start:
        start, end = end, start
        
    fbxSDKExport.set_fps(fps)
    # animations for these object types
    ob_anim_lists = ob_bones, ob_meshes, ob_arms

    if use_anim and [tmp for tmp in ob_anim_lists if tmp]:

        frame_orig = scene.frame_current

        if use_anim_optimize:
            # Do we really want to keep such behavior? User could enter real value directly...
            ANIM_OPTIMIZE_PRECISSION_FLOAT = 10 ** (-anim_optimize_precision + 2)

        # default action, when no actions are avaioable
        tmp_actions = []
        blenActionDefault = None
        action_lastcompat = None

        # instead of tagging
        tagged_actions = []

        # get the current action first so we can use it if we only export one action (JCB)
        for my_arm in ob_arms:
            blenActionDefault = my_arm.blenAction
            if blenActionDefault:
                break

        if use_anim_action_all:
            tmp_actions = bpy.data.actions[:]
        elif not use_default_take:
            if blenActionDefault:
                # Export the current action (JCB)
                tmp_actions.append(blenActionDefault)

        if tmp_actions:
            # find which actions are compatible with the armatures
            tmp_act_count = 0
            for my_arm in ob_arms:

                arm_bone_names = set([my_bone.blenName for my_bone in my_arm.fbxBones])

                for action in tmp_actions:

                    if arm_bone_names.intersection(action_bone_names(my_arm.blenObject, action)):  # at least one channel matches.
                        my_arm.blenActionList.append(action)
                        tagged_actions.append(action.name)
                        tmp_act_count += 1

                        # in case there are no actions applied to armatures
                        # for example, when a user deletes the current action.
                        action_lastcompat = action

            if tmp_act_count:
                # unlikely to ever happen but if no actions applied to armatures, just use the last compatible armature.
                if not blenActionDefault:
                    blenActionDefault = action_lastcompat

        del action_lastcompat

        if use_default_take:
            tmp_actions.insert(0, None)  # None is the default action
               
        for blenAction in tmp_actions:
            # we have tagged all actious that are used be selected armatures
            if blenAction:
                if blenAction.name in tagged_actions:
                    print('\taction: "%s" exporting...' % blenAction.name)
                else:
                    print('\taction: "%s" has no armature using it, skipping' % blenAction.name)
                    continue

            if blenAction is None:
                # Warning, this only accounts for tmp_actions being [None]
                take_name = "Default Take"
                act_start = start
                act_end = end
            else:
                # use existing name
                take_name = sane_name_mapping_take.get(blenAction.name)
                if take_name is None:
                    take_name = sane_takename(blenAction)

                act_start, act_end = blenAction.frame_range
                act_start = int(act_start)
                act_end = int(act_end)

                # Set the action active
                for my_arm in ob_arms:
                    if my_arm.blenObject.animation_data and blenAction in my_arm.blenActionList:
                        my_arm.blenObject.animation_data.action = blenAction

            # Use the action name as the take name and the take filename (JCB)
            #fw('\n\tTake: "%s" {' % take_name)
            # set pose data for all bones
            # do this here in case the action changes
            fbxSDKExport.set_time_span(c_char_p(take_name.encode('utf-8')), act_start - 1, act_end - 1, act_start - 1, act_end - 1)
            '''
            for my_bone in ob_bones:
                my_bone.flushAnimData()
            '''
            i = act_start
            while i <= act_end:
                scene.frame_set(i)
                for ob_generic in ob_anim_lists:
                    for my_ob in ob_generic:
                        #Blender.Window.RedrawAll()
                        if ob_generic == ob_meshes and my_ob.fbxArm:
                            # We cant animate armature meshes!
                            my_ob.setPoseFrame(i, fake=True)
                        else:
                            my_ob.setPoseFrame(i)

                i += 1

            #for bonename, bone, obname, me, armob in ob_bones:
            for ob_generic in (ob_bones, ob_meshes, ob_arms):

                for my_ob in ob_generic:

                    if ob_generic == ob_meshes and my_ob.fbxArm:
                        # do nothing,
                        pass
                    else:

                        #fw('\n\t\tModel: "Model::%s" {' % my_ob.fbxName)
                        
                        context_bone_anim_mats = [(my_ob.getAnimParRelMatrix(frame), my_ob.getAnimParRelMatrixRot(frame)) for frame in range(act_start, act_end + 1)]

                        # ----------------
                        # ----------------
                        for TX_LAYER, TX_CHAN in enumerate('TRS'):  # transform, rotate, scale

                            if TX_CHAN == 'T':
                                context_bone_anim_vecs = [mtx[0].to_translation() for mtx in context_bone_anim_mats]
                            elif    TX_CHAN == 'S':
                                context_bone_anim_vecs = [mtx[0].to_scale() for mtx in context_bone_anim_mats]
                            elif    TX_CHAN == 'R':
                                # Was....
                                # elif     TX_CHAN=='R':    context_bone_anim_vecs = [mtx[1].to_euler()            for mtx in context_bone_anim_mats]
                                #
                                # ...but we need to use the previous euler for compatible conversion.
                                context_bone_anim_vecs = []
                                prev_eul = None
                                for mtx in context_bone_anim_mats:
                                    if prev_eul:
                                        prev_eul = mtx[1].to_euler('XYZ', prev_eul)
                                    else:
                                        prev_eul = mtx[1].to_euler()
                                    context_bone_anim_vecs.append(tuple_rad_to_deg(prev_eul))

                            #fw('\n\t\t\t\tChannel: "%s" {' % TX_CHAN)  # translation

                            for i in range(3):
                                # Loop on each axis of the bone
                                #fw('\n\t\t\t\t\tChannel: "%s" {' % ('XYZ'[i]))  # translation
                                #fw('\n\t\t\t\t\t\tDefault: %.15f' % context_bone_anim_vecs[0][i])
                                fbxSDKExport.set_channel_default_value(c_char_p(take_name.encode('utf-8')), c_char_p(my_ob.fbxName.encode('utf-8')), to_channel_type(TX_CHAN, i), context_bone_anim_vecs[0][i])
                                if not use_anim_optimize:
                                    # Just write all frames, simple but in-eficient
                                    frame = act_start
                                    while frame <= act_end:
                                        # Curve types are 'C,n' for constant, 'L' for linear
                                        # C,n is for bezier? - linear is best for now so we can do simple keyframe removal
                                        #fw('\n\t\t\t\t\t\t\t%i,%.15f,L' % (fbx_time(frame - 1), context_bone_anim_vecs[frame - act_start][i]))
                                        fbxSDKExport.add_channel_key(c_char_p(take_name.encode('utf-8')), c_char_p(my_ob.fbxName.encode('utf-8')), 
                                                                     to_channel_type(TX_CHAN, i), frame - 1, context_bone_anim_vecs[frame - act_start][i])
                                        frame += 1
                                else:
                                    # remove unneeded keys, j is the frame, needed when some frames are removed.
                                    context_bone_anim_keys = [(vec[i], j) for j, vec in enumerate(context_bone_anim_vecs)]

                                    # last frame to fisrt frame, missing 1 frame on either side.
                                    # removeing in a backwards loop is faster
                                    #for j in xrange( (act_end-act_start)-1, 0, -1 ):
                                    # j = (act_end-act_start)-1
                                    j = len(context_bone_anim_keys) - 2
                                    while j > 0 and len(context_bone_anim_keys) > 2:
                                        # print j, len(context_bone_anim_keys)
                                        # Is this key the same as the ones next to it?

                                        # co-linear horizontal...
                                        if        abs(context_bone_anim_keys[j][0] - context_bone_anim_keys[j - 1][0]) < ANIM_OPTIMIZE_PRECISSION_FLOAT and \
                                                abs(context_bone_anim_keys[j][0] - context_bone_anim_keys[j + 1][0]) < ANIM_OPTIMIZE_PRECISSION_FLOAT:

                                            del context_bone_anim_keys[j]

                                        else:
                                            frame_range = float(context_bone_anim_keys[j + 1][1] - context_bone_anim_keys[j - 1][1])
                                            frame_range_fac1 = (context_bone_anim_keys[j + 1][1] - context_bone_anim_keys[j][1]) / frame_range
                                            frame_range_fac2 = 1.0 - frame_range_fac1

                                            if abs(((context_bone_anim_keys[j - 1][0] * frame_range_fac1 + context_bone_anim_keys[j + 1][0] * frame_range_fac2)) - context_bone_anim_keys[j][0]) < ANIM_OPTIMIZE_PRECISSION_FLOAT:
                                                del context_bone_anim_keys[j]
                                            else:
                                                j -= 1

                                        # keep the index below the list length
                                        if j > len(context_bone_anim_keys) - 2:
                                            j = len(context_bone_anim_keys) - 2

                                    if len(context_bone_anim_keys) == 2 and context_bone_anim_keys[0][0] == context_bone_anim_keys[1][0]:

                                        # This axis has no moton, its okay to skip KeyCount and Keys in this case
                                        # pass

                                        # better write one, otherwise we loose poses with no animation
                                        #fw('\n\t\t\t\t\t\t\t%i,%.15f,L' % (fbx_time(start), context_bone_anim_keys[0][0]))
                                        fbxSDKExport.add_channel_key(c_char_p(take_name.encode('utf-8')), c_char_p(my_ob.fbxName.encode('utf-8')), 
                                                                     to_channel_type(TX_CHAN, i), start, context_bone_anim_keys[0][0])
                                    else:
                                        # We only need to write these if there is at least one
                                        for val, frame in context_bone_anim_keys:
                                            # frame is already one less then blenders frame
                                            #fw('\n\t\t\t\t\t\t\t%i,%.15f,L' % (fbx_time(frame), val))
                                            fbxSDKExport.add_channel_key(c_char_p(take_name.encode('utf-8')), c_char_p(my_ob.fbxName.encode('utf-8')), 
                                                                     to_channel_type(TX_CHAN, i), frame, val)

            # end action loop. set original actions
            # do this after every loop in case actions effect eachother.
            for my_arm in ob_arms:
                if my_arm.blenObject.animation_data:
                    my_arm.blenObject.animation_data.action = my_arm.blenAction
                    
        scene.frame_set(frame_orig)                    
    
    # write meshes animation
    #for obname, ob, mtx, me, mats, arm, armname in ob_meshes:

    # Clear mesh data Only when writing with modifiers applied
    for me in meshes_to_clear:
        bpy.data.meshes.remove(me)    
    
    # XXX, shouldnt be global!
    for mapping in (sane_name_mapping_ob,
                    sane_name_mapping_ob_unique,
                    sane_name_mapping_mat,
                    sane_name_mapping_tex,
                    sane_name_mapping_take,
                    sane_name_mapping_group,
                    ):
        mapping.clear()
    del mapping
    
    
    del ob_arms[:]
    del ob_bones[:]
    del ob_meshes[:]
    
    #fbxSDKExport.print_skeleton()
    fbxSDKExport.print_mesh()
    #fbxSDKExport.print_takes()
    
    fbxSDKExport.export(c_char_p(filepath.encode('utf-8')))
    
    # Clear cached ObjectWrappers!
    ObjectWrapper.cache_clear()    
    
    print('export sdk finished in %.4f sec.' % (time.process_time() - start_time))    
    
    return {'FINISHED'}


def save(operator, context,
         filepath="",
         use_selection=False,
         batch_mode='OFF',
         use_batch_own_dir=False,
         **kwargs
         ):
    
    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')
        
    kwargs_mod = kwargs.copy()
    kwargs_mod["context_objects"] = context.scene.objects
    
    return save_single(operator, context.scene, filepath, **kwargs_mod)


