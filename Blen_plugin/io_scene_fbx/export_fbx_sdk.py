import os
import time
import math  # math.pi
from ctypes import c_char_p

import bpy
from mathutils import Vector, Matrix
from print_util import PrintHelper

from fbx_export import FBXExport

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

def save_single(operator, scene, filepath="",
        global_matrix=None,
        context_objects=None,
        object_types={'EMPTY', 'CAMERA', 'LAMP', 'ARMATURE', 'MESH'},
        use_mesh_modifiers=True,
        mesh_smooth_type='FACE',
        use_armature_deform_only=False,
        use_anim=True,
        use_anim_optimize=True,
        anim_optimize_precision=6,
        use_anim_action_all=False,
        use_metadata=True,
        path_mode='AUTO',
        use_mesh_edges=True,
        use_default_take=True,
        **kwargs
    ):
    
    print("export_fbx_sdk::save_single")
    
    # Used for mesh and armature rotations
    mtx4_z90 = Matrix.Rotation(math.pi / 2.0, 4, 'Z')
    
    if global_matrix is None:
        global_matrix = Matrix()
        global_scale = 1.0
    else:
        global_scale = global_matrix.median_scale    
    
    fbxSDKExport = FBXExport(5)
    
    
    
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

            # Lamps need to be rotated
            if obj_type == 'LAMP':
                matrix_rot = matrix_rot * mtx_x90
            elif obj_type == 'CAMERA':
                y = matrix_rot * Vector((0.0, 1.0, 0.0))
                matrix_rot = Matrix.Rotation(math.pi / 2.0, 3, y) * matrix_rot

            return matrix_rot

    # ----------------------------------------------
    
    def write_mesh(my_mesh):
        me = my_mesh.blenData
        
        fbxSDKExport.set_mesh_name(c_char_p(my_mesh.fbxName.encode('utf-8')))
        
        _nchunk = 3
        t_co = [None] * len(me.vertices) * 3
        me.vertices.foreach_get("co", t_co)
        for vertex in grouper_exact(t_co, _nchunk):
            fbxSDKExport.add_vertex(vertex[0], vertex[1], vertex[2])
        
        t_vi = [None] * len(me.loops)
        me.loops.foreach_get("vertex_index", t_vi)
        for ix in t_vi:
            fbxSDKExport.add_index(ix)
        
        t_ls = [None] * len(me.polygons)
        me.polygons.foreach_get("loop_start", t_ls)
        for ls in t_ls:
            fbxSDKExport.add_loop_start(ls)
        
        if t_ls != sorted(t_ls):
            print("Error: polygons and loops orders do not match!")
            
        t_vn = [None] * len(me.loops) * 3
        me.calc_normals_split()
        me.loops.foreach_get("normal", t_vn)
        for vnormal in grouper_exact(t_vn, _nchunk):
            fbxSDKExport.add_normal(vnormal[0], vnormal[1], vnormal[2])
            
        del t_vn
        me.free_normals_split()
        del t_co
        del t_vi
        del t_ls
        
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
            t_pi = None
            uv2idx = None
            tex2idx = None
            
            if do_textures:
                is_tex_unique = len(my_mesh.blenTextures) == 1
                tex2idx = {None: -1}
                tex2idx.update({tex: i for i, tex in enumerate(my_mesh.blenTextures)})       
                
            for uvindex, (uvlayer, uvtexture) in enumerate(zip(uvlayers, uvtextures)):
                uvlayer.data.foreach_get("uv", t_uv)
                uvco = tuple(zip(*[iter(t_uv)] * 2))
                fbxSDKExport.create_uv_info(uvindex, c_char_p(uvlayer.name.encode('utf-8')))
        
        fbxSDKExport.Print()
        # add meshes here to clear because they are not used anywhere.
    meshes_to_clear = []
    
    ob_meshes = []
    
    groups = []  # blender groups, only add ones that have objects in the selections
    materials = set()  # (mat, image) items
    textures = set()
        
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
    
        
    # sanity checks
    try:
        assert(not (ob_meshes and ('MESH' not in object_types)))
        
    except AssertionError:
        import traceback
        traceback.print_exc()
        
    for my_mesh in ob_meshes:
        write_mesh(my_mesh)
        
    fbxSDKExport.export(c_char_p(filepath.encode('utf-8')))
    
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