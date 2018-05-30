#include "Foo.h"
#include "Common/Common.h"
#include <cmath>

Foo::Foo(int n)
{
	val = n;
	mAsASCII = false;
	mLogFile = std::ofstream("ExportFBXSdk.log");
	mVertices = std::vector<Foo::Vector3>();
	mNormals = std::vector<Foo::Vector3>();
	mIndices = std::vector<int>();
	mLoopStart = std::vector<int>();
	mSmoothing = std::vector<int>();
	mUVInfos = std::map<int, LayerElementUVInfo>();
	mMaterials = std::vector<Material>();
	mBones = std::map<std::string, Bone>();
	mDeformers = std::map<std::string, Deformer>();
	mPoses = std::map<std::string, Pose>();
	mTakes = std::map<std::string, Take>();
	mTextures = std::map<std::string, Texture>();
	mCoutbuf = std::cout.rdbuf(); //save old buf
	std::cout.rdbuf(mLogFile.rdbuf()); //redirect std::cout to out.txt!
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

	for (std::pair<std::string, Mesh> _mesh : mMesh)
	{
		const Mesh& mesh = _mesh.second;
		FbxMesh* lMesh = FbxMesh::Create(pScene, mesh.mMeshName.c_str());

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

		FbxLayer* lLayer = lMesh->GetLayer(0);
		if (lLayer == NULL)
		{
			lMesh->CreateLayer();
			lLayer = lMesh->GetLayer(0);
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
			lLayer->SetUVs(lUVElement, FbxLayerElement::eUV);
		}

		FbxGeometryElementMaterial* lMaterialElement = lMesh->CreateElementMaterial();
		lMaterialElement->SetReferenceMode(FbxGeometryElement::eIndexToDirect);
		lMaterialElement->SetMappingMode(FbxGeometryElement::eByPolygon);
		if (mesh.mMatIndices.size() == 0)
		{
			lMaterialElement->SetMappingMode(FbxGeometryElement::eAllSame);
			lMaterialElement->GetIndexArray().Add(0);
		}
		lLayer->SetMaterials(lMaterialElement);

		for (int i = 0; i < mLoopStart.size(); ++i)
		{
			int loopStart = mLoopStart[i];
			size_t loopEnd = i < mLoopStart.size() - 1 ? mLoopStart[i + 1] - 1 : mIndices.size() - 1;
			lMesh->BeginPolygon(mesh.mMatIndices.size() == 0 ? -1 : mesh.mMatIndices[i]);
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

		lMesh->BeginAddMeshEdgeIndex();
		for (const IntVector2& edge : mesh.edges)
		{
			lMesh->AddMeshEdgeIndex(edge.x, edge.y, false);
		}
		lMesh->EndAddMeshEdgeIndex();

		FbxNode* lRootNode = pScene->GetRootNode();
		FbxNode* lNode = FbxNode::Create(pScene, mesh.mMeshName.c_str());
		lNode->LclTranslation.Set(FbxVector4(mesh.lclTranslation.x, mesh.lclTranslation.y, mesh.lclTranslation.z));
		lNode->LclRotation.Set(FbxVector4(mesh.lclRotation.x, mesh.lclRotation.y, mesh.lclRotation.z));
		lNode->LclScaling.Set(FbxVector4(mesh.lclScaling.x, mesh.lclScaling.y, mesh.lclScaling.z));
		lRootNode->AddChild(lNode);
		lNode->SetNodeAttribute(lMesh);
		pMeshNode = lNode;

		std::map<std::string, FbxSurfacePhong*> materials = std::map<std::string, FbxSurfacePhong*>();
		for (const Material& mat : mMaterials)
		{
			FbxSurfacePhong *lMaterial = FbxSurfacePhong::Create(pScene, FbxString(mat.materialName.c_str()).Buffer());
			lMaterial->ShadingModel.Set(FbxString(mat.shadingName.c_str()));
			lMaterial->Emissive.Set(mat.emissiveColor);
			lMaterial->Ambient.Set(mat.ambientColor);
			lMaterial->Diffuse.Set(mat.diffuseColor);
			FbxNode* lNode = lMesh->GetNode();
			if (lNode)
				lNode->AddMaterial(lMaterial);
			materials.insert(make_pair(mat.materialName, lMaterial));
		}

		for (std::pair<std::string, Texture> _tex : mTextures)
		{
			const Texture& tex = _tex.second;
			FbxFileTexture* lTexture = FbxFileTexture::Create(pScene, tex.name.c_str());
			lTexture->SetFileName(tex.fileName.c_str());
			lTexture->SetRelativeFileName(tex.relFileName.c_str());
			lTexture->SetAlphaSource(static_cast<FbxTexture::EAlphaSource>(tex.alphaSource));
			lTexture->SetPremultiplyAlpha(tex.premultiplyAlpha);
			if (tex.currentMappingType == 0)
			{
				lTexture->SetMappingType(FbxTexture::EMappingType::eUV);
			}
			else if (tex.currentMappingType == 1)
			{
				lTexture->SetMappingType(FbxTexture::EMappingType::ePlanar);
			}
			else if (tex.currentMappingType == 2)
			{
				lTexture->SetMappingType(FbxTexture::EMappingType::eSpherical);
			}
			else if (tex.currentMappingType == 3)
			{
				lTexture->SetMappingType(FbxTexture::EMappingType::eCylindrical);
			}
			else if (tex.currentMappingType == 4)
			{
				lTexture->SetMappingType(FbxTexture::EMappingType::eBox);
			}
			lTexture->UVSet.Set(FbxString(tex.UVSet.c_str()));
			lTexture->SetWrapMode(static_cast<FbxTexture::EWrapMode>(tex.wrapModeU), static_cast<FbxTexture::EWrapMode>(tex.wrapModeV));
			lTexture->SetTranslation(tex.translation.x, tex.translation.y);
			lTexture->SetScale(tex.scaling.x, tex.scaling.y);
			lTexture->UseMaterial.Set(tex.useMaterial);
			lTexture->UseMipMap.Set(tex.useMipMap);

			for (const std::string& parentMat : tex.parentMat)
			{
				if (FbxSurfacePhong* lMaterialPhong = materials.at(parentMat))
				{
					if (tex.matProp == std::string("DiffuseColor"))
					{
						lMaterialPhong->Diffuse.ConnectSrcObject(lTexture);
					}
					else if (tex.matProp == std::string("NormalMap"))
					{
						lMaterialPhong->NormalMap.ConnectSrcObject(lTexture);
					}
				}
			}
		}
	}



	return true;
}

bool Foo::BuildPose(FbxScene* pScene, FbxNode*& pMeshNode, FbxNode* pSkeletonNode)
{
	if (pMeshNode == nullptr || pSkeletonNode == nullptr)
	{
		return true;
	}

	FbxPose* lPose = FbxPose::Create(pScene, pMeshNode->GetName());
	lPose->SetIsBindPose(true);

	for (std::pair<std::string, Pose> _pose : mPoses)
	{
		const Pose& pose = _pose.second;
		Mat4x4 transform = pose.poseTransform;
		FbxMatrix lBindMatrix = FbxMatrix(transform.x0, transform.x1, transform.x2, transform.x3,
			transform.y0, transform.y1, transform.y2, transform.y3,
			transform.z0, transform.z1, transform.z2, transform.z3,
			transform.w0, transform.w1, transform.w2, transform.w3);

		FbxNode*  lKFbxNode = nullptr;
		if (pose.poseNodeName == std::string(pSkeletonNode->GetName()))
		{
			lKFbxNode = pSkeletonNode;
		}
		else if (pose.poseNodeName == std::string(pMeshNode->GetName()))
		{
			lKFbxNode = pMeshNode;
		}
		else
		{
			lKFbxNode = pSkeletonNode->FindChild(pose.poseNodeName.c_str());
		}

		if (lKFbxNode == nullptr)
		{
			return false;
		}

		lPose->Add(lKFbxNode, lBindMatrix);
	}

	pScene->AddPose(lPose);

	return true;
}

bool Foo::BuildTakes(FbxScene* pScene, FbxNode* pSkeletonNode)
{
	if (pSkeletonNode == nullptr)
	{
		return true;
	}

	FbxTime::SetGlobalTimeMode(FbxTime::eCustom, mFps);
	for (std::pair<std::string, Take> _take : mTakes)
	{
		const Take& take = _take.second;
		FbxAnimStack* lAnimStack = FbxAnimStack::Create(pScene, take.takeName.c_str());
		FbxAnimLayer* lAnimLayer = FbxAnimLayer::Create(pScene, "Base Layer");
		lAnimStack->AddMember(lAnimLayer);
		FbxTime start, end;
		start.SetFramePrecise(take.localTimeSpan[0], FbxTime::eCustom);
		end.SetFramePrecise(take.localTimeSpan[1], FbxTime::eCustom);
		lAnimStack->SetLocalTimeSpan(FbxTimeSpan(start, end));

		start.SetFramePrecise(take.referenceTimeSpan[0], FbxTime::eCustom);
		end.SetFramePrecise(take.referenceTimeSpan[1], FbxTime::eCustom);

		lAnimStack->SetReferenceTimeSpan(FbxTimeSpan(start, end));

		for (std::pair<std::string, ModelAnim> _model : take.models)
		{
			const ModelAnim& model = _model.second;
			FbxNode*  lKFbxNode = nullptr;
			if (model.modelName == std::string(pSkeletonNode->GetName()))
			{
				lKFbxNode = pSkeletonNode;
			}
			else
			{
				lKFbxNode = pSkeletonNode->FindChild(model.modelName.c_str());
			}

			if (lKFbxNode == nullptr)
			{
				return false;
			}

			FbxAnimCurveNode* animTranslationCurveNode = lKFbxNode->LclTranslation.GetCurveNode(lAnimLayer, true);
			FbxAnimCurveNode* animRotationCurveNode = lKFbxNode->LclRotation.GetCurveNode(lAnimLayer, true);
			FbxAnimCurveNode* animScalingCurveNode = lKFbxNode->LclScaling.GetCurveNode(lAnimLayer, true);
			if (animTranslationCurveNode == nullptr || animRotationCurveNode == nullptr || animScalingCurveNode == nullptr)
			{
				return false;
			}

			FbxAnimCurve* lLclTranslationXCurve = lKFbxNode->LclTranslation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_X, true);
			FbxAnimCurve* lLclTranslationYCurve = lKFbxNode->LclTranslation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_Y, true);
			FbxAnimCurve* lLclTranslationZCurve = lKFbxNode->LclTranslation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_Z, true);

			FbxAnimCurve* lLclRotationXCurve = lKFbxNode->LclRotation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_X, true);
			FbxAnimCurve* lLclRotationYCurve = lKFbxNode->LclRotation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_Y, true);
			FbxAnimCurve* lLclRotationZCurve = lKFbxNode->LclRotation.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_Z, true);

			FbxAnimCurve* lLclScalingXCurve = lKFbxNode->LclScaling.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_X, true);
			FbxAnimCurve* lLclScalingYCurve = lKFbxNode->LclScaling.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_Y, true);
			FbxAnimCurve* lLclScalingZCurve = lKFbxNode->LclScaling.GetCurve(lAnimLayer, FBXSDK_CURVENODE_COMPONENT_Z, true);

			if (lLclTranslationXCurve == nullptr || lLclTranslationYCurve == nullptr || lLclTranslationZCurve == nullptr
				|| lLclRotationXCurve == nullptr || lLclRotationYCurve == nullptr || lLclRotationZCurve == nullptr
				|| lLclScalingXCurve == nullptr || lLclScalingYCurve == nullptr || lLclScalingZCurve == nullptr)
			{
				return false;
			}

			FillDefaultValueKeys(animTranslationCurveNode, lLclTranslationXCurve, FBXSDK_CURVENODE_COMPONENT_X, model.channels[T_X]);
			FillDefaultValueKeys(animTranslationCurveNode, lLclTranslationYCurve, FBXSDK_CURVENODE_COMPONENT_Y, model.channels[T_Y]);
			FillDefaultValueKeys(animTranslationCurveNode, lLclTranslationZCurve, FBXSDK_CURVENODE_COMPONENT_Z, model.channels[T_Z]);

			FillDefaultValueKeys(animRotationCurveNode, lLclRotationXCurve, FBXSDK_CURVENODE_COMPONENT_X, model.channels[R_X]);
			FillDefaultValueKeys(animRotationCurveNode, lLclRotationYCurve, FBXSDK_CURVENODE_COMPONENT_Y, model.channels[R_Y]);
			FillDefaultValueKeys(animRotationCurveNode, lLclRotationZCurve, FBXSDK_CURVENODE_COMPONENT_Z, model.channels[R_Z]);

			FillDefaultValueKeys(animScalingCurveNode, lLclScalingXCurve, FBXSDK_CURVENODE_COMPONENT_X, model.channels[S_X]);
			FillDefaultValueKeys(animScalingCurveNode, lLclScalingYCurve, FBXSDK_CURVENODE_COMPONENT_Y, model.channels[S_Y]);
			FillDefaultValueKeys(animScalingCurveNode, lLclScalingZCurve, FBXSDK_CURVENODE_COMPONENT_Z, model.channels[S_Z]);
		}
	}

	return true;
}

bool Foo::FillDefaultValueKeys(FbxAnimCurveNode* animCurveNode, FbxAnimCurve* animCurve, const char* channelName, const Channel& channel)
{
	animCurveNode->SetChannelValue(channelName, channel.defaultValue);

	animCurve->KeyModifyBegin();
	FbxTime lTime;
	int lKeyIndex = 0;

	for (const Key& key : channel.keys)
	{
		lTime.SetFramePrecise(key.frame, FbxTime::EMode::eCustom);
		lKeyIndex = animCurve->KeyAdd(lTime);
		animCurve->KeySetValue(lKeyIndex, key.value);
		animCurve->KeySetInterpolation(lKeyIndex, FbxAnimCurveDef::eInterpolationCubic);
	}
	animCurve->KeyModifyEnd();

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

	if (!BuildDeformer(pScene, pMeshNode, pSkeletonNode != nullptr ? pSkeletonNode->GetChild(0) : nullptr))
	{
		return false;
	}

	if (!BuildPose(pScene, pMeshNode, pSkeletonNode))
	{
		return false;
	}

	if (!BuildTakes(pScene, pSkeletonNode))
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

Foo::Mesh& Foo::GetMesh(char* name)
{
	std::string meshName = std::string(name);
	std::map<std::string, Mesh>::iterator ite = mMesh.find(meshName);
	if (ite == mMesh.end())
	{
		mMesh.insert(make_pair(std::string(name), Mesh(name)));
	}

	return mMesh.at(meshName);
}

void Foo::AddMatIndex(char* name, int index)
{
	Mesh& mesh = GetMesh(name);
	mesh.mMatIndices.push_back(index);
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

void Foo::AddMeshEdge(char* name, int startVertexIndex, int endVertexIndex)
{
	Mesh& mesh = GetMesh(name);
	mesh.edges.push_back(IntVector2(startVertexIndex, endVertexIndex));
}

void Foo::SetMeshProperty(char* name, Vector3 trans, Vector3 rot, Vector3 sca)
{
	Mesh& mesh = GetMesh(name);
	mesh.mMeshName = std::string(name);
	mesh.lclTranslation = trans;
	mesh.lclRotation = rot;
	mesh.lclScaling = sca;
}

void Foo::AddMaterial(char* mName, char* sName, Vector3 diffuse, Vector3 ambient, Vector3 emissive)
{
	Material mat = Material(mName, sName, diffuse, ambient, emissive);
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

void Foo::SetTextureMatProp(char* name, char* matName, char* matProp)
{
	Texture& tex = mTextures.at(std::string(name));
	tex.parentMat.push_back(matName);
	tex.matProp = std::string(matProp);
}

void Foo::AddTexture(char* name, char* fileName, char* relFileName, int alphaSource, bool premultiplyAlpha, int currentMappingType, char* UVSet, int wrapModeU, int wrapModeV,
	Vector3 translation, Vector3 scaling, bool useMaterial, bool useMipMap)
{
	mTextures.insert(make_pair(std::string(name), Texture(name, fileName, relFileName, alphaSource, premultiplyAlpha, currentMappingType, UVSet, wrapModeU, wrapModeV,
		translation, scaling, useMaterial, useMipMap)));
}

void Foo::AddPoseNode(char* name, Mat4x4 transform)
{
	mPoses.insert(make_pair(std::string(name), Pose(name, transform)));
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

Foo::ModelAnim& Foo::GetModelAnim(char* tName, char* mName)
{
	std::string takeName = std::string(tName);
	std::string modelName = std::string(mName);

	std::map<std::string, Take>::iterator ite = mTakes.find(takeName);
	if (ite == mTakes.end())
	{
		mTakes.insert(make_pair(takeName, Take(tName)));
	}

	std::map<std::string, ModelAnim>& models = mTakes.at(takeName).models;
	std::map<std::string, ModelAnim>::iterator ite2 = models.find(modelName);
	if (ite2 == models.end())
	{
		models.insert(make_pair(modelName, ModelAnim(mName)));
	}

	return models.at(modelName);
}

void Foo::SetTimeSpan(char* tName, float lStart, float lEnd, float rStart, float rEnd)
{
	std::string takeName = std::string(tName);

	std::map<std::string, Take>::iterator ite = mTakes.find(takeName);
	if (ite == mTakes.end())
	{
		mTakes.insert(make_pair(takeName, Take(tName)));
	}

	Take& take = mTakes.at(takeName);
	take.localTimeSpan[0] = lStart;
	take.localTimeSpan[1] = lEnd;

	take.referenceTimeSpan[0] = rStart;
	take.referenceTimeSpan[1] = rEnd;
}

void Foo::SetChannelDefaultValue(char* takeName, char* modelName, int type, double value)
{
	if (type >= ChannelType::ChannelMax)
	{
		return;
	}

	ChannelType channelType = static_cast<ChannelType>(type);
	ModelAnim& model = GetModelAnim(takeName, modelName);
	
	model.channels[channelType].defaultValue = value;
}

void Foo::AddChannelKey(char* takeName, char* modelName, int type, float frame, float value)
{
	if (type >= ChannelType::ChannelMax)
	{
		return;
	}

	ChannelType channelType = static_cast<ChannelType>(type);
	ModelAnim& model = GetModelAnim(takeName, modelName);

	model.channels[channelType].keys.push_back(Key(frame, value));
}

void Foo::SetFPS(float fps)
{
	mFps = fps;
}

void Foo::PrintTakes()
{
	std::cout << "FPS: " << mFps << std::endl;
	for (std::pair<std::string, Take> _take : mTakes)
	{
		const Take& take = _take.second;
		std::cout << "take name: " << take.takeName << std::endl;
		std::cout << "local time span: [" << take.localTimeSpan[0] << ", " << take.localTimeSpan[1] << "]" << std::endl;
		std::cout << "reference time span: [" << take.referenceTimeSpan[0] << ", " << take.referenceTimeSpan[1] << "]" << std::endl;
		for (std::pair<std::string, ModelAnim> _model : take.models)
		{
			ModelAnim model = _model.second;
			std::cout << "model name: " << model.modelName << std::endl;
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
	}
}

void Foo::PrintSkeleton()
{
	for (std::pair<std::string, Bone> _bone : mBones)
	{
		const Bone& bone = _bone.second;
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

	for (std::pair<std::string, Pose> _pose : mPoses)
	{
		const Pose& pose = _pose.second;
		std::cout << "pose node name: " << pose.poseNodeName << " transform:" << std::endl
			<< pose.poseTransform << std::endl;
	}
}

void Foo::PrintMesh()
{
	for (Foo::Vector3 v : mVertices)
	{
		std::cout << "vertex[ " << v << " ]" << std::endl;
	}

	for (std::pair<std::string, Mesh> _mesh : mMesh)
	{
		const Mesh& mesh = _mesh.second;

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

		std::cout << "mesh name: " << mesh.mMeshName << std::endl;
		std::cout << "mesh translation: " << mesh.lclTranslation << " rotation: " << mesh.lclRotation << " scale: " << mesh.lclScaling << std::endl;

		for (const Foo::IntVector2& edge : mesh.edges)
		{
			std::cout << "edge" << edge << std::endl;
		}

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
		for (int ix : mesh.mMatIndices)
		{
			std::cout << ix << ", ";
		}
		std::cout << " ]" << std::endl;

		for (const Material& mat : mMaterials)
		{
			std::cout << "Material [material name: " << mat.materialName << ", shading name: " << mat.shadingName << "]" << std::endl;
			std::cout << "diffuse color: " << mat.diffuseColor[0] << ", " << mat.diffuseColor[1] << ", " << mat.diffuseColor[2] << std::endl;
			std::cout << "ambient color: " << mat.ambientColor[0] << ", " << mat.ambientColor[1] << ", " << mat.ambientColor[2] << std::endl;
			std::cout << "emissive color: " << mat.emissiveColor[0] << ", " << mat.emissiveColor[1] << ", " << mat.emissiveColor[2] << std::endl;
		}

		std::cout << "Textures:" << std::endl;
		for (std::pair<std::string, Texture> _tex : mTextures)
		{
			const Texture& tex = _tex.second;
			std::cout << "name: " << tex.name << " filename: " << tex.fileName << " rel filename: " << tex.relFileName << std::endl << " alphaSource: " << tex.alphaSource << " premultiplyAlpha: " << tex.premultiplyAlpha
				<< " currentMappingType: " << tex.currentMappingType << " UVSet: " << tex.UVSet << " wrapModeU: " << tex.wrapModeU << " wrapModeV: " << tex.wrapModeV << std::endl << " translation: " << tex.translation
				<< " scaling: " << tex.scaling << " useMaterial: " << tex.useMaterial << " useMipMap: " << tex.useMipMap << " mat Prop: " << tex.matProp << std::endl;
			std::cout << "mat Parent: " << std::endl;
			for (const std::string& parentMat : tex.parentMat)
			{
				std::cout << parentMat << ", " << std::endl;
			}

		}
	}

	
}

bool Foo::Export(char* filePath)
{
	std::cout.rdbuf(mCoutbuf); //reset to standard output again
	mLogFile.flush();
	mLogFile.close();

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
	lResult = SaveScene(lSdkManager, lScene, filePath, mAsASCII);

	if (lResult == false)
	{
		PrintString("\n\nAn error occurred while saving the scene...\n");
		DestroySdkObjects(lSdkManager, lResult);
		return false;
	}

	return true;
}

std::ostream& operator<< (std::ostream &os, const Foo::IntVector2& vec)
{
	os << "[ " << vec.x << ", " << vec.y << "]";
	return os;
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
	DLLEXPORT void Foo_AddMatIndex(Foo* foo, char* name, int n) { foo->AddMatIndex(name, n); }
	DLLEXPORT void Foo_AddUVIndex(Foo* foo, int uvIndex, int n) { foo->AddUVIndex(uvIndex, n); }
	DLLEXPORT void Foo_AddLoopStart(Foo* foo, int n) { foo->AddLoopStart(n); }
	DLLEXPORT void Foo_AddMeshEdge(Foo* foo, char* name, int s, int e) { foo->AddMeshEdge(name, s, e); }
	DLLEXPORT void Foo_AddSmoothing(Foo* foo, int s) { foo->AddSmoothing(s); }
	DLLEXPORT void Foo_SetSmoothMode(Foo* foo, int m) { foo->SetSmoothMode(m); }
	DLLEXPORT void Foo_SetMeshProperty(Foo* foo, char* name, Foo::Vector3 trans, Foo::Vector3 rot, Foo::Vector3 sca) 
		{ foo->SetMeshProperty(name, trans, rot, sca); }
	DLLEXPORT void Foo_AddMaterial(Foo* foo, char* mName, char* sName, Foo::Vector3 diffuse, Foo::Vector3 ambient, Foo::Vector3 emissive) 
	{ foo->AddMaterial(mName, sName, diffuse, ambient, emissive); }
	DLLEXPORT void Foo_AddTexture(Foo* foo, char* name, char* fileName, char* relFileName, int alphaSource, bool premultiplyAlpha, int currentMappingType, char* UVSet, int wrapModeU, int wrapModeV,
		Foo::Vector3 translation, Foo::Vector3 scaling, bool useMaterial, bool useMipMap) { foo->AddTexture(name, fileName, relFileName, alphaSource, premultiplyAlpha, currentMappingType, UVSet, 
			wrapModeU, wrapModeV, translation, scaling, useMaterial, useMipMap);}
	DLLEXPORT void Foo_SetTextureMatProp(Foo* foo, char* name, char* matName, char* matProp) { foo->SetTextureMatProp(name, matName, matProp); }

	DLLEXPORT void Foo_AddPoseNode(Foo* foo, char* name, Foo::Mat4x4 mat) { foo->AddPoseNode(name, mat); }
	DLLEXPORT void Foo_AddBoneChild(Foo* foo, char* cName, char* pName) { foo->AddBoneChild(cName, pName); }
	DLLEXPORT void Foo_AddBone(Foo* foo, char* name, Foo::Vector3 lclTranslation, Foo::Vector3 lclRotation, Foo::Vector3 lclScaling)
		{ foo->AddBone(name, lclTranslation, lclRotation, lclScaling); }
	DLLEXPORT void Foo_AddSubDeformerIndex(Foo* foo, char* mName, char* bName, int index) { foo->AddSubDeformerIndex(mName, bName, index); }
	DLLEXPORT void Foo_AddSubDeformerWeight(Foo* foo, char* mName, char* bName, float weight) { foo->AddSubDeformerWeight(mName, bName, weight); }
	DLLEXPORT void Foo_SetSubDeformerTransform(Foo* foo, char* mName, char* bName, Foo::Mat4x4 transf, Foo::Vector4 quat) { foo->SetSubDeformerTransform(mName, bName, transf, quat); }
	DLLEXPORT void Foo_SetSubDeformerTransformLink(Foo* foo, char* mName, char* bName, Foo::Mat4x4 transfLink) { foo->SetSubDeformerTransformLink(mName, bName, transfLink); }

	DLLEXPORT void Foo_SetFPS(Foo* foo, float fps) { foo->SetFPS(fps); }
	DLLEXPORT void Foo_SetTimeSpan(Foo* foo, char* tName, float lStart, float lEnd, float rStart, float rEnd) { foo->SetTimeSpan(tName, lStart, lEnd, rStart, rEnd); }
	DLLEXPORT void Foo_SetChannelDefaultValue(Foo* foo, char* takeName, char* modelName, int type, double value) { foo->SetChannelDefaultValue(takeName, modelName, type, value); }
	DLLEXPORT void Foo_AddChannelKey(Foo* foo, char* takeName, char* modelName, int type, float frame, float value) { foo->AddChannelKey(takeName, modelName, type, frame, value); }

	DLLEXPORT bool Foo_Export(Foo* foo, char* name) { return foo->Export(name); }
	DLLEXPORT void Foo_PrintMesh(Foo* foo) { foo->PrintMesh(); }
	DLLEXPORT void Foo_PrintSkeleton(Foo* foo) { foo->PrintSkeleton(); }
	DLLEXPORT void Foo_PrintTakes(Foo* foo) { foo->PrintTakes(); }

	DLLEXPORT void Foo_SetAsASCII(Foo* foo, bool asAscii) { foo->SetAsASCII(asAscii); }
}
