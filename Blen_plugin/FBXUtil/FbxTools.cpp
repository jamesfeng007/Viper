#include "FbxTools.h"

namespace FBXUtil
{
	Mesh& GetMesh(char* name, std::map<std::string, Mesh>& meshLoader)
	{
		std::string meshName = std::string(name);
		std::map<std::string, Mesh>::iterator ite = meshLoader.find(meshName);
		if (ite == meshLoader.end())
		{
			meshLoader.insert(make_pair(std::string(name), Mesh(name)));
		}

		return meshLoader.at(meshName);
	}

	LayerElementUVInfo& GetUVInfo(int uvIndex, const char* name, Mesh& mesh)
	{
		std::map<int, LayerElementUVInfo>::iterator ite = mesh.mUVInfos.find(uvIndex);
		if (ite == mesh.mUVInfos.end())
		{
			LayerElementUVInfo uvInfo = LayerElementUVInfo(uvIndex, name);
			mesh.mUVInfos.insert(std::make_pair(uvIndex, uvInfo));
		}
		return mesh.mUVInfos.at(uvIndex);
	}

	void PrintModelAnim(const ModelAnim& model)
	{
		std::cout << "model name: " << model.modelName <<" ref bone: " <<model.refModelUUID << " ref stack: " 
			<< model.refStackUUID << " ref layer: " << model.refLayerUUID<< std::endl;
		for (int i = ChannelType::T_X; i < ChannelType::ChannelMax; ++i)
		{
			switch (i)
			{
			case ChannelType::T_X:
				std::cout << "channel T_X ";
				break;
			case ChannelType::T_Y:
				std::cout << "channel T_Y ";
				break;
			case ChannelType::T_Z:
				std::cout << "channel T_Z ";
				break;
			case ChannelType::R_X:
				std::cout << "channel R_X ";
				break;
			case ChannelType::R_Y:
				std::cout << "channel R_Y ";
				break;
			case ChannelType::R_Z:
				std::cout << "channel R_Z ";
				break;
			case ChannelType::S_X:
				std::cout << "channel S_X ";
				break;
			case ChannelType::S_Y:
				std::cout << "channel S_Y ";
				break;
			case ChannelType::S_Z:
				std::cout << "channel S_Z ";
				break;
			default:
				break;
			}
			std::cout << "default value: " << model.channels[i].defaultValue << std::endl;
			std::cout << "keys: [";
			for (Key key : model.channels[i].keys)
			{
				std::cout << "(" << key.frame << ", " << key.value << "), ";
			}
			std::cout << "]" << std::endl;
		}
	}

	void PrintNode(const Node& node)
	{
		std::cout << "node name: " << node.nodeName << " is bone: " << node.isBone << " attribute name: " << node.nodeAttributeName << std::endl;
		std::cout << "node translation: " << node.lclTranslation << " rotation: " << node.lclRotation << " scale: " << node.lclScaling << std::endl;
		std::cout << "node GeometricTranslation: " << node.GeometricTranslation << " GeometricRotation: " << node.GeometricRotation << " GeometricScaling: " << node.GeometricScaling << std::endl;
		std::cout << "node RotationOffset: " << node.RotationOffset << " RotationPivot: " << node.RotationPivot << " ScalingOffset: " << node.ScalingOffset << std::endl;
		std::cout << "node ScalingPivot: " << node.ScalingPivot << " PreRotation: " << node.PreRotation << " PostRotation: " << node.PostRotation << std::endl;
		std::cout << "node RotationOrder: " << node.RotationOrder << " RotationActive: " << node.RotationActive << std::endl;
	}

	void PrintPoseNode(const PoseNode& node)
	{
		std::cout << "pose node name: " << node.poseNodeName <<", ref node uuid: " << node.refNodeUuid << ", transform:" << std::endl
			<< node.poseTransform << std::endl;
	}

	void PrintSubDeformer(const SubDeformer& subDeformer)
	{
		std::cout << "subdeformer name: " << subDeformer.subDeformerName << " uuid: " << subDeformer.uuid << " link bone uuid: "
			<< subDeformer.linkBoneUuid << std::endl;

		std::cout << "subdeformer index[ ";
		for (int ix : subDeformer.indexes)
		{
			std::cout << ix << ", ";
		}
		std::cout << " ]" << std::endl;

		std::cout << "subdeformer weight[ ";
		for (double ix : subDeformer.weights)
		{
			std::cout << ix << ", ";
		}
		std::cout << " ]" << std::endl;

		std::cout << "transform:" << std::endl;
		std::cout << subDeformer.transform << std::endl;
		std::cout << "transformLink:" << std::endl;
		std::cout << subDeformer.transformLink << std::endl;
	}

	void PrintMaterial(const Material& mat)
	{
		std::cout << "Material [material name: " << mat.materialName << ", shading name: " << mat.shadingName << "]" << std::endl;
		std::cout << "diffuse color: " << mat.diffuseColor[0] << ", " << mat.diffuseColor[1] << ", " << mat.diffuseColor[2] << std::endl;
		std::cout << "ambient color: " << mat.ambientColor[0] << ", " << mat.ambientColor[1] << ", " << mat.ambientColor[2] << std::endl;
		std::cout << "emissive color: " << mat.emissiveColor[0] << ", " << mat.emissiveColor[1] << ", " << mat.emissiveColor[2] << std::endl;
	}

	void PrintTexture(const Texture& tex)
	{
		std::cout << "name: " << tex.name << " filename: " << tex.fileName << " rel filename: " << tex.relFileName << std::endl << " alphaSource: " << tex.alphaSource << " premultiplyAlpha: " << tex.premultiplyAlpha
			<< " currentMappingType: " << tex.currentMappingType << " UVSet: " << tex.UVSet << " wrapModeU: " << tex.wrapModeU << " wrapModeV: " << tex.wrapModeV << std::endl << " translation: " << tex.translation
			<< " scaling: " << tex.scaling << " rotation: " << tex.rotation << " useMaterial: " << tex.useMaterial << " useMipMap: " << tex.useMipMap << " mat Prop: " << tex.matProp << std::endl;
		std::cout << "mat Parent: " << std::endl;
		for (const std::string& parentMat : tex.parentMat)
		{
			std::cout << parentMat << ", " << std::endl;
		}
	}

	void PrintMesh(const Mesh& mesh)
	{
		std::cout << "mesh name: " << mesh.mMeshName << std::endl;

		for (Vector3 v : mesh.mVertices)
		{
			std::cout << "vertex[ " << v << " ]" << std::endl;
		}

		std::cout << "index[ ";
		for (int ix : mesh.mIndices)
		{
			std::cout << ix << ", ";
		}
		std::cout << " ]" << std::endl;

		for (Vector3 n : mesh.mNormals)
		{
			std::cout << "normal[ " << n << " ]" << std::endl;
		}

		for (const Vector3& tan : mesh.mTangents)
		{
			std::cout << "tangent[ " << tan << " ]" << std::endl;
		}

		for (const Vector3& bino : mesh.mBinormals)
		{
			std::cout << "binormal[ " << bino << " ]" << std::endl;
		}

		std::cout << "start[ ";
		for (int s : mesh.mLoopStart)
		{
			std::cout << s << ", ";
		}
		std::cout << " ]" << std::endl;

		for (const IntVector2& edge : mesh.edges)
		{
			std::cout << "edge" << edge << std::endl;
		}

		std::cout << "smoothing mode:" << mesh.mSmoothMode << std::endl;
		for (int s : mesh.mSmoothing)
		{
			std::cout << s << ", ";
		}
		std::cout << std::endl;

		for (std::pair<int, LayerElementUVInfo> _uvInfo : mesh.mUVInfos)
		{
			LayerElementUVInfo uvInfo = _uvInfo.second;
			std::cout << "uv Index: " << uvInfo.uvIndex << " name: " << uvInfo.name << std::endl;
			for (UV uv : uvInfo.mUVs)
			{
				std::cout << "uv[ " << uv.x << ", " << uv.y << "]" << std::endl;
			}

			std::cout << "UVIndex[ ";
			for (int ix : uvInfo.mUVIndices)
			{
				std::cout << ix << ", ";
			}
			std::cout << " ]" << std::endl;
		}

		std::cout << "MatIndex[ ";
		for (int ix : mesh.mMatIndices)
		{
			std::cout << ix << ", ";
		}
		std::cout << " ]" << std::endl;
	}

	void PrintBone(const Bone& bone)
	{
		std::cout << "Bone name: " << bone.boneName << " translation: " << bone.lclTranslation << " rotation: "
			<< bone.lclRotation << " scaling: " << bone.lclScaling << " parent: " << bone.parentName << std::endl;
	}

	std::ostream& operator<< (std::ostream &os, const IntVector2& vec)
	{
		os << "[ " << vec.x << ", " << vec.y << "]";
		return os;
	}

	std::ostream& operator<< (std::ostream &os, const Vector3& vec)
	{
		os << "[ " << vec.x << ", " << vec.y << ", " << vec.z << "]";
		return os;
	}

	std::ostream& operator<< (std::ostream &os, const Mat4x4& mat)
	{
		os << "[ " << mat.x0 << ", " << mat.x1 << ", " << mat.x2 << ", " << mat.x3 << "]" << std::endl
			<< "[ " << mat.y0 << ", " << mat.y1 << ", " << mat.y2 << ", " << mat.y3 << "]" << std::endl
			<< "[ " << mat.z0 << ", " << mat.z1 << ", " << mat.z2 << ", " << mat.z3 << "]" << std::endl
			<< "[ " << mat.w0 << ", " << mat.w1 << ", " << mat.w2 << ", " << mat.w3 << "]";
		return os;
	}
}
