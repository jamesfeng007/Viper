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

	void PrintMesh(const Mesh& mesh)
	{
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

		std::cout << "mesh name: " << mesh.mMeshName << std::endl;
		std::cout << "mesh translation: " << mesh.lclTranslation << " rotation: " << mesh.lclRotation << " scale: " << mesh.lclScaling << std::endl;
		std::cout << "mesh GeometricTranslation: " << mesh.GeometricTranslation << " GeometricRotation: " << mesh.GeometricRotation << " GeometricScaling: " << mesh.GeometricScaling << std::endl;
		std::cout << "mesh RotationOffset: " << mesh.RotationOffset << " RotationPivot: " << mesh.RotationPivot << " ScalingOffset: " << mesh.ScalingOffset << std::endl;
		std::cout << "mesh ScalingPivot: " << mesh.ScalingPivot << " PreRotation: " << mesh.PreRotation << " PostRotation: " << mesh.PostRotation << std::endl;
		std::cout << "mesh RotationOrder: " << mesh.RotationOrder << " RotationActive: " << mesh.RotationActive << std::endl;

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
