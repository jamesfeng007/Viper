#include "Foo.h"
#include "Common/Common.h"
#include <cmath>

Foo::Foo(int n)
{
	val = n;
	mVertices = std::vector<Foo::Vector3>();
	mNormals = std::vector<Foo::Vector3>();
	mIndices = std::vector<int>();
	mLoopStart = std::vector<int>();
	mSmoothing = std::vector<int>();
	mUVInfos = std::map<int, LayerElementUVInfo>();
	mMatIndices = std::vector<int>();
	mMaterials = std::vector<Material>();
	mBones = std::map<std::string, Bone>();
	mDeformers = std::map<std::string, Deformer>();
}

bool Foo::BuildArmature(FbxScene* pScene, FbxNode*& pSkeletonNode)
{
	std::map<std::string, FbxNode*> reg = std::map<std::string, FbxNode*>();

	for (std::pair<std::string, Bone> _bone : mBones)
	{
		Bone bone = _bone.second;
		FbxNode* pNode = FbxNode::Create(pScene, bone.boneName.c_str());
		FbxSkeleton* lSkeleton = FbxSkeleton::Create(pScene, bone.boneName.c_str());
		lSkeleton->SetSkeletonType(FbxSkeleton::eLimbNode);
		pNode->SetNodeAttribute(lSkeleton);
		pNode->LclTranslation.Set(FbxVector4(bone.lclTranslation.x, bone.lclTranslation.y, bone.lclTranslation.z));
		pNode->LclRotation.Set(FbxVector4(bone.lclRotation.x, bone.lclRotation.y, bone.lclRotation.z));
		pNode->LclScaling.Set(FbxVector4(bone.lclScaling.x, bone.lclScaling.y, bone.lclScaling.z));
		reg.insert(make_pair(bone.boneName, pNode));
	}

	for (std::pair<std::string, Bone> _bone : mBones)
	{
		FbxNode* pChild = reg.at(_bone.second.boneName);
		if (!_bone.second.parentName.empty())
		{
			FbxNode* pParent = reg.at(_bone.second.parentName);
			pParent->AddChild(pChild);
		}
		else
		{
			pScene->GetRootNode()->AddChild(pChild);
			pSkeletonNode = pChild;
		}
	}

	return true;
}

bool Foo::FillDeformer(FbxScene* pScene, FbxNode* pMeshNode, FbxNode* pSkeletonNode, FbxSkin* lSkin)
{
	if (pMeshNode == nullptr || pSkeletonNode == nullptr)
	{
		return false;
	}

	FbxCluster *lCluster = FbxCluster::Create(pScene, std::string(pSkeletonNode->GetName()).append("_Cluster").c_str());
	lCluster->SetLink(pSkeletonNode);
	lCluster->SetLinkMode(FbxCluster::eTotalOne);

	SubDeformer subDeformer = GetSubDeformer(pMeshNode->GetName(), pSkeletonNode->GetName());
	if (subDeformer.indexes.size() != subDeformer.weights.size())
	{
		return false;
	}

	size_t indexWeightCount = subDeformer.indexes.size();
	for (int i = 0; i < indexWeightCount; ++i)
	{
		lCluster->AddControlPointIndex(subDeformer.indexes.at(i), subDeformer.weights.at(i));
	}

	Mat4x4 transform = subDeformer.transform;
	FbxMatrix transf = FbxMatrix(transform.x0, transform.x1, transform.x2, transform.x3,
		transform.y0, transform.y1, transform.y2, transform.y3,
		transform.z0, transform.z1, transform.z2, transform.z3,
		transform.w0, transform.w1, transform.w2, transform.w3);
	FbxVector4 pTranslation;
	FbxVector4 pRotation;
	FbxVector4 pShearing;
	FbxVector4 pScaling;
	double pSign;
	transf.GetElements(pTranslation, pRotation, pShearing, pScaling, pSign);
	FbxAMatrix aTrans = FbxAMatrix(pTranslation, pRotation, pScaling);
	lCluster->SetTransformMatrix(aTrans);

	transform = subDeformer.transformLink;
	transf = FbxMatrix(transform.x0, transform.x1, transform.x2, transform.x3,
		transform.y0, transform.y1, transform.y2, transform.y3,
		transform.z0, transform.z1, transform.z2, transform.z3,
		transform.w0, transform.w1, transform.w2, transform.w3);
	transf.GetElements(pTranslation, pRotation, pShearing, pScaling, pSign);
	aTrans = FbxAMatrix(pTranslation, pRotation, pScaling);
	lCluster->SetTransformLinkMatrix(aTrans);

	lSkin->AddCluster(lCluster);

	int childCount = pSkeletonNode->GetChildCount();
	for (int i = 0; i < childCount; ++i)
	{
		FillDeformer(pScene, pMeshNode, pSkeletonNode->GetChild(i), lSkin);
	}

	return true;
}

bool Foo::BuildDeformer(FbxScene* pScene, FbxNode* pMeshNode, FbxNode* pSkeletonNode)
{
	if (pMeshNode == nullptr || pSkeletonNode == nullptr)
	{
		return true;
	}

	FbxGeometry* lMeshAttribute = (FbxGeometry*)pMeshNode->GetNodeAttribute();
	FbxSkin* lSkin = FbxSkin::Create(pScene, "lSkin");

	if (!FillDeformer(pScene, pMeshNode, pSkeletonNode, lSkin))
	{
		return false;
	}

	lMeshAttribute->AddDeformer(lSkin);
	return true;
}

bool Foo::BuildMesh(FbxScene* pScene, FbxNode*& pMeshNode)
{
	if (mVertices.empty())
	{
		return true;
	}

	FbxMesh* lMesh = FbxMesh::Create(pScene, mMesh.mMeshName.c_str());

	lMesh->InitControlPoints(static_cast<int>(mVertices.size()));
	FbxVector4* lControlPoints = lMesh->GetControlPoints();
	for (int i = 0; i < mVertices.size(); ++i)
	{
		lMesh->SetControlPointAt(FbxVector4(mVertices[i].x, mVertices[i].y, mVertices[i].z), i);
	}

	FbxGeometryElementNormal* lGeometryElementNormal = lMesh->CreateElementNormal();
	lGeometryElementNormal->SetMappingMode(FbxGeometryElement::eByPolygonVertex);
	lGeometryElementNormal->SetReferenceMode(FbxGeometryElement::eDirect);
	for (int i = 0; i < mNormals.size(); ++i)
	{
		lGeometryElementNormal->GetDirectArray().Add(FbxVector4(mNormals[i].x, mNormals[i].y, mNormals[i].z));
	}

	for (std::pair<int, Foo::LayerElementUVInfo> _uvInfo : mUVInfos)
	{
		Foo::LayerElementUVInfo uvInfo = _uvInfo.second;
		FbxGeometryElementUV* lUVElement = lMesh->CreateElementUV(uvInfo.name.c_str());
		lUVElement->SetMappingMode(FbxGeometryElement::eByPolygonVertex);
		lUVElement->SetReferenceMode(FbxGeometryElement::eIndexToDirect);
		for (UV uv : uvInfo.mUVs)
		{
			lUVElement->GetDirectArray().Add(FbxVector2(uv.x, uv.y));
		}
		lUVElement->GetIndexArray().SetCount(0);
		for (int uvIndex : uvInfo.mUVIndices)
		{
			lUVElement->GetIndexArray().Add(uvIndex);
		}
	}

	FbxGeometryElementMaterial* lMaterialElement = lMesh->CreateElementMaterial();
	lMaterialElement->SetReferenceMode(FbxGeometryElement::eIndexToDirect);
	lMaterialElement->SetMappingMode(FbxGeometryElement::eByPolygon);
	if (mMatIndices.size() == 0)
	{
		lMaterialElement->SetMappingMode(FbxGeometryElement::eAllSame);
		lMaterialElement->GetIndexArray().Add(0);
	}

	for (int i = 0; i < mLoopStart.size(); ++i)
	{
		int loopStart = mLoopStart[i];
		size_t loopEnd = i < mLoopStart.size() - 1 ? mLoopStart[i + 1] - 1 : mIndices.size() - 1;
		lMesh->BeginPolygon(mMatIndices.size() == 0 ? -1 : mMatIndices[i]);
		for (int j = loopStart; j <= loopEnd; ++j)
		{
			lMesh->AddPolygon(mIndices[j]);
		}
		lMesh->EndPolygon();
	}

	FbxGeometryElementSmoothing* lSmoothingElement = nullptr;
	if (mSmoothMode >= 0)
	{
		lSmoothingElement = lMesh->CreateElementSmoothing();
		lSmoothingElement->SetReferenceMode(FbxLayerElement::eDirect);
	}

	switch (mSmoothMode)
	{
	case 0:
		lSmoothingElement->SetMappingMode(FbxLayerElement::eByPolygon);
		break;
	case 1:
		lSmoothingElement->SetMappingMode(FbxLayerElement::eByEdge);
		break;
	default:
		mSmoothing.clear();
	}

	if (lSmoothingElement != nullptr)
	{
		for (int smoothingFlag : mSmoothing)
		{
			lSmoothingElement->GetDirectArray().Add(smoothingFlag);
		}
	}

	FbxNode* lRootNode = pScene->GetRootNode();
	FbxNode* lNode = FbxNode::Create(pScene, mMesh.mMeshName.c_str());
	lNode->LclTranslation.Set(FbxVector4(mMesh.lclTranslation.x, mMesh.lclTranslation.y, mMesh.lclTranslation.z));
	lNode->LclRotation.Set(FbxVector4(mMesh.lclRotation.x, mMesh.lclRotation.y, mMesh.lclRotation.z));
	lNode->LclScaling.Set(FbxVector4(mMesh.lclScaling.x, mMesh.lclScaling.y, mMesh.lclScaling.z));
	lRootNode->AddChild(lNode);
	lNode->SetNodeAttribute(lMesh);
	pMeshNode = lNode;

	for (Material mat : mMaterials)
	{
		FbxSurfacePhong *lMaterial = FbxSurfacePhong::Create(pScene, FbxString(mat.materialName.c_str()).Buffer());
		lMaterial->ShadingModel.Set(FbxString(mat.shadingName.c_str()));
		FbxNode* lNode = lMesh->GetNode();
		if (lNode)
			lNode->AddMaterial(lMaterial);
	}

	return true;
}

bool Foo::CreateScene(FbxScene* pScene)
{
	FbxNode* pMeshNode = nullptr;
	FbxNode* pSkeletonNode = nullptr;

	if (!BuildMesh(pScene, pMeshNode))
	{
		return false;
	}

	if (!BuildArmature(pScene, pSkeletonNode))
	{
		return false;
	}

	if (!BuildDeformer(pScene, pMeshNode, pSkeletonNode))
	{
		return false;
	}

	return true;
}

void Foo::CreateUVInfo(int uvIndex, char* name)
{
	Foo::LayerElementUVInfo uvInfo = LayerElementUVInfo(uvIndex, name);
	mUVInfos.insert(std::make_pair(uvIndex, uvInfo));
}

void Foo::AddVertex(float x, float y, float z)
{
	mVertices.push_back(Foo::Vector3(x, y, z));
}

void Foo::AddNormal(float x, float y, float z)
{
	mNormals.push_back(Foo::Vector3(x, y, z));
}

void Foo::AddUV(int uvIndex, float x, float y)
{
	mUVInfos.at(uvIndex).mUVs.push_back(Foo::UV(x, y));
}

void Foo::AddUVIndex(int uvIndex, int index)
{
	mUVInfos.at(uvIndex).mUVIndices.push_back(index);
}

void Foo::AddMatIndex(int index)
{
	mMatIndices.push_back(index);
}

void Foo::AddIndex(int index)
{
	mIndices.push_back(index);
}

void Foo::AddLoopStart(int start)
{
	mLoopStart.push_back(start);
}

void Foo::AddSmoothing(int smooth)
{
	mSmoothing.push_back(smooth);
}

void Foo::SetSmoothMode(int mode)
{
	mSmoothMode = mode;
}

void Foo::SetMeshProperty(char* name, Vector3 trans, Vector3 rot, Vector3 sca)
{
	mMesh.mMeshName = std::string(name);
	mMesh.lclTranslation = trans;
	mMesh.lclRotation = rot;
	mMesh.lclScaling = sca;
}

void Foo::AddMaterial(char* mName, char* sName)
{
	Material mat = Material(mName, sName);
	mMaterials.push_back(mat);
}

void Foo::AddBone(char* name, Vector3 lclTranslation, Vector3 lclRotation, Vector3 lclScaling)
{
	mBones.insert(make_pair(std::string(name), Bone(name, lclTranslation, lclRotation, lclScaling)));
}

void Foo::AddBoneChild(char* child, char* parent)
{
	Bone& parentB = mBones.at(std::string(parent));
	Bone& childB = mBones.at(std::string(child));
	childB.parentName = parentB.boneName;
}

Foo::SubDeformer& Foo::GetSubDeformer(const char* mName, const char* bName)
{
	std::string meshName = std::string(mName);
	std::string boneName = std::string(bName);

	std::map<std::string, Deformer>::iterator ite = mDeformers.find(meshName);
	if (ite == mDeformers.end())
	{
		mDeformers.insert(make_pair(meshName, Deformer(mName)));
	}

	std::map<std::string, SubDeformer>& subDeformers = mDeformers.at(meshName).subDeformers;
	std::map<std::string, SubDeformer>::iterator ite2 = subDeformers.find(boneName);
	if (ite2 == subDeformers.end())
	{
		subDeformers.insert(make_pair(boneName, SubDeformer(mName, bName)));
	}

	return subDeformers.at(boneName);
}

void Foo::SetSubDeformerTransformLink(char* mName, char* bName, Mat4x4 transformLink)
{
	SubDeformer& subDeformer = GetSubDeformer(mName, bName);
	subDeformer.transformLink = transformLink;
}

void Foo::SetSubDeformerTransform(char* mName, char* bName, Mat4x4 transform, Vector4 quat)
{
	SubDeformer& subDeformer = GetSubDeformer(mName, bName);
	subDeformer.transform = transform;
	subDeformer.quat = quat;
}

void Foo::AddSubDeformerWeight(char* mName, char* bName, float weight)
{
	SubDeformer& subDeformer = GetSubDeformer(mName, bName);
	subDeformer.weights.push_back(weight);
}

void Foo::AddSubDeformerIndex(char* mName, char* bName, int index)
{
	SubDeformer& subDeformer = GetSubDeformer(mName, bName);
	subDeformer.indexes.push_back(index);
	
}

void Foo::PrintSkeleton()
{
	for (std::pair<std::string, Foo::Bone> _bone : mBones)
	{
		const Foo::Bone& bone = _bone.second;
		std::cout << "Bone name: " << _bone.first << " translation: " << bone.lclTranslation << " rotation: " 
			<< bone.lclRotation << " scaling: " << bone.lclScaling << " parent: " << bone.parentName << std::endl;
	}

	for (std::pair<std::string, Deformer> _deformer : mDeformers)
	{
		const Deformer& deformer = _deformer.second;
		std::cout << "deformer name: " << deformer.deformerName << std::endl;
		for (std::pair<std::string, SubDeformer> _subDeformer : deformer.subDeformers)
		{
			const SubDeformer& subDeformer = _subDeformer.second;
			std::cout << "subdeformer name: " << subDeformer.subDeformerName << std::endl;

			std::cout << "subdeformer index[ ";
			for (int ix : subDeformer.indexes)
			{
				std::cout << ix << ", ";
			}
			std::cout << " ]" << std::endl;

			std::cout << "subdeformer weight[ ";
			for (float ix : subDeformer.weights)
			{
				std::cout << ix << ", ";
			}
			std::cout << " ]" << std::endl;

			std::cout << "transform:" << std::endl;
			std::cout << subDeformer.transform << std::endl;
			std::cout << "transformLink:" << std::endl;
			std::cout << subDeformer.transformLink << std::endl;
		}
	}
}

void Foo::PrintMesh()
{
	for (Foo::Vector3 v : mVertices)
	{
		std::cout << "vertex[ " << v << " ]" << std::endl;
	}

	std::cout << "index[ ";
	for (int ix : mIndices)
	{
		std::cout << ix << ", ";
	}
	std::cout << " ]" << std::endl;

	for (Foo::Vector3 n : mNormals)
	{
		std::cout << "normal[ " << n << " ]" << std::endl;
	}

	std::cout << "start[ ";
	for (int s : mLoopStart)
	{
		std::cout << s << ", ";
	}
	std::cout << " ]" << std::endl;

	std::cout << "mesh name: " << mMesh.mMeshName << std::endl;
	std::cout << "mesh translation: " << mMesh.lclTranslation << " rotation: " << mMesh.lclRotation << " scale: " << mMesh.lclScaling << std::endl;

	std::cout << "smoothing mode:" << mSmoothMode << std::endl;
	for (int s : mSmoothing)
	{
		std::cout << s << ", ";
	}
	std::cout << std::endl;

	for (std::pair<int, Foo::LayerElementUVInfo> _uvInfo : mUVInfos)
	{
		Foo::LayerElementUVInfo uvInfo = _uvInfo.second;
		std::cout << "uv Index: " << uvInfo.uvIndex << " name: " << uvInfo.name << std::endl;
		for (Foo::UV uv : uvInfo.mUVs)
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
	for (int ix : mMatIndices)
	{
		std::cout << ix << ", ";
	}
	std::cout << " ]" << std::endl;

	for (Foo::Material mat : mMaterials)
	{
		std::cout << "Material [material name: " << mat.materialName << ", shading name: " << mat.shadingName << "]" << std::endl;
	}
}

bool Foo::Export(char* filePath)
{
	FbxManager* lSdkManager = NULL;
	FbxScene* lScene = NULL;

	InitializeSdkObjects(lSdkManager, lScene);
	bool lResult = CreateScene(lScene);
	if (lResult == false)
	{
		PrintString("\n\nAn error occurred while creating the scene...\n");
		DestroySdkObjects(lSdkManager, lResult);
		return false;
	}

	// Save the scene.
	lResult = SaveScene(lSdkManager, lScene, filePath);

	if (lResult == false)
	{
		PrintString("\n\nAn error occurred while saving the scene...\n");
		DestroySdkObjects(lSdkManager, lResult);
		return false;
	}

	return true;
}

std::ostream& operator<< (std::ostream &os, const Foo::Vector3& vec)
{
	os << "[ " << vec.x << ", " << vec.y << ", " << vec.z << "]";
	return os;
}

std::ostream& operator<< (std::ostream &os, const Foo::Mat4x4& mat)
{
	os << "[ " << mat.x0 << ", " << mat.x1 << ", " << mat.x2 << ", " << mat.x3 << "]" << std::endl
		<< "[ " << mat.y0 << ", " << mat.y1 << ", " << mat.y2 << ", " << mat.y3 << "]" << std::endl
		<< "[ " << mat.z0 << ", " << mat.z1 << ", " << mat.z2 << ", " << mat.z3 << "]" << std::endl
		<< "[ " << mat.w0 << ", " << mat.w1 << ", " << mat.w2 << ", " << mat.w3 << "]";
	return os;
}

extern "C"
{
	DLLEXPORT Foo* Foo_new(int n) { return new Foo(n); }
	DLLEXPORT void Foo_AddVertex(Foo* foo, float x, float y, float z) { foo->AddVertex(x, y, z); }
	DLLEXPORT void Foo_AddNormal(Foo* foo, float x, float y, float z) { foo->AddNormal(x, y, z); }
	DLLEXPORT void Foo_CreateUVInfo(Foo* foo, int uvIndex, char* name) { foo->CreateUVInfo(uvIndex, name); }
	DLLEXPORT void Foo_AddUV(Foo* foo, int uvIndex, float x, float y) { foo->AddUV(uvIndex, x, y); }
	DLLEXPORT void Foo_AddIndex(Foo* foo, int n) { foo->AddIndex(n); }
	DLLEXPORT void Foo_AddMatIndex(Foo* foo, int n) { foo->AddMatIndex(n); }
	DLLEXPORT void Foo_AddUVIndex(Foo* foo, int uvIndex, int n) { foo->AddUVIndex(uvIndex, n); }
	DLLEXPORT void Foo_AddLoopStart(Foo* foo, int n) { foo->AddLoopStart(n); }
	DLLEXPORT void Foo_AddSmoothing(Foo* foo, int s) { foo->AddSmoothing(s); }
	DLLEXPORT void Foo_SetSmoothMode(Foo* foo, int m) { foo->SetSmoothMode(m); }
	DLLEXPORT void Foo_SetMeshProperty(Foo* foo, char* name, Foo::Vector3 trans, Foo::Vector3 rot, Foo::Vector3 sca) 
		{ foo->SetMeshProperty(name, trans, rot, sca); }
	DLLEXPORT void Foo_AddMaterial(Foo* foo, char* mName, char* sName) { foo->AddMaterial(mName, sName); }
	DLLEXPORT void Foo_AddBoneChild(Foo* foo, char* cName, char* pName) { foo->AddBoneChild(cName, pName); }
	DLLEXPORT void Foo_AddBone(Foo* foo, char* name, Foo::Vector3 lclTranslation, Foo::Vector3 lclRotation, Foo::Vector3 lclScaling)
		{ foo->AddBone(name, lclTranslation, lclRotation, lclScaling); }
	DLLEXPORT void Foo_AddSubDeformerIndex(Foo* foo, char* mName, char* bName, int index) { foo->AddSubDeformerIndex(mName, bName, index); }
	DLLEXPORT void Foo_AddSubDeformerWeight(Foo* foo, char* mName, char* bName, float weight) { foo->AddSubDeformerWeight(mName, bName, weight); }
	DLLEXPORT void Foo_SetSubDeformerTransform(Foo* foo, char* mName, char* bName, Foo::Mat4x4 transf, Foo::Vector4 quat) { foo->SetSubDeformerTransform(mName, bName, transf, quat); }
	DLLEXPORT void Foo_SetSubDeformerTransformLink(Foo* foo, char* mName, char* bName, Foo::Mat4x4 transfLink) { foo->SetSubDeformerTransformLink(mName, bName, transfLink); }
	DLLEXPORT bool Foo_Export(Foo* foo, char* name) { return foo->Export(name); }
	DLLEXPORT void Foo_PrintMesh(Foo* foo) { foo->PrintMesh(); }
	DLLEXPORT void Foo_PrintSkeleton(Foo* foo) { foo->PrintSkeleton(); }
}
