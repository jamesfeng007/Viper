#include "Exporter.h"
#include "Common/Common.h"
#include <cmath>

Exporter::Exporter()
{
	mAsASCII = false;
	mLogFile = std::ofstream("ExportFBXSdk.log");
	mMesh = std::map<std::string, Mesh>();
	mMaterials = std::vector<Material>();
	mBones = std::map<std::string, Bone>();
	mDeformers = std::map<std::string, Deformer>();
	mPoses = std::map<std::string, PoseNode>();
	mTakes = std::map<std::string, Take>();
	mTextures = std::map<std::string, Texture>();
	mCoutbuf = std::cout.rdbuf(); //save old buf
	std::cout.rdbuf(mLogFile.rdbuf()); //redirect std::cout to out.txt!
}

bool Exporter::BuildArmature(FbxScene* pScene, FbxNode*& pSkeletonNode)
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

bool Exporter::FillDeformer(FbxScene* pScene, FbxNode* pMeshNode, FbxNode* pSkeletonNode, FbxSkin* lSkin)
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

bool Exporter::BuildDeformer(FbxScene* pScene, FbxNode* pMeshNode, FbxNode* pSkeletonNode)
{
	if (pMeshNode == nullptr || pSkeletonNode == nullptr)
	{
		return true;
	}

	if (mDeformers.size() == 0)
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

bool Exporter::BuildMesh(FbxScene* pScene, FbxNode*& pMeshNode)
{
	for (std::pair<std::string, Mesh> _mesh : mMesh)
	{
		Mesh& mesh = _mesh.second;

		if (mesh.mVertices.empty())
		{
			continue;
		}

		FbxMesh* lMesh = FbxMesh::Create(pScene, mesh.mMeshName.c_str());

		lMesh->InitControlPoints(static_cast<int>(mesh.mVertices.size()));
		FbxVector4* lControlPoints = lMesh->GetControlPoints();
		for (int i = 0; i < mesh.mVertices.size(); ++i)
		{
			lMesh->SetControlPointAt(FbxVector4(mesh.mVertices[i].x, mesh.mVertices[i].y, mesh.mVertices[i].z), i);
		}

		FbxGeometryElementNormal* lGeometryElementNormal = lMesh->CreateElementNormal();
		lGeometryElementNormal->SetMappingMode(FbxGeometryElement::eByPolygonVertex);
		lGeometryElementNormal->SetReferenceMode(FbxGeometryElement::eDirect);
		for (int i = 0; i < mesh.mNormals.size(); ++i)
		{
			lGeometryElementNormal->GetDirectArray().Add(FbxVector4(mesh.mNormals[i].x, mesh.mNormals[i].y, mesh.mNormals[i].z));
		}

		if (mesh.mTangents.size() > 0)
		{
			FbxGeometryElementTangent* lGeometryElementTangent = lMesh->CreateElementTangent();
			lGeometryElementTangent->SetName(mesh.tangentName.c_str());
			lGeometryElementTangent->SetMappingMode(FbxLayerElement::eByPolygonVertex);
			lGeometryElementTangent->SetReferenceMode(FbxLayerElement::eDirect);
			for (int i = 0; i < mesh.mTangents.size(); ++i)
			{
				lGeometryElementTangent->GetDirectArray().Add(FbxVector4(mesh.mTangents[i].x, mesh.mTangents[i].y, mesh.mTangents[i].z));
			}
		}

		if (mesh.mBinormals.size() > 0)
		{
			FbxGeometryElementBinormal* lGeometryElementBinormal = lMesh->CreateElementBinormal();
			lGeometryElementBinormal->SetName(mesh.binormalName.c_str());
			lGeometryElementBinormal->SetMappingMode(FbxLayerElement::eByPolygonVertex);
			lGeometryElementBinormal->SetReferenceMode(FbxLayerElement::eDirect);
			for (int i = 0; i < mesh.mBinormals.size(); ++i)
			{
				lGeometryElementBinormal->GetDirectArray().Add(FbxVector4(mesh.mBinormals[i].x, mesh.mBinormals[i].y, mesh.mBinormals[i].z));
			}
		}
		

		FbxLayer* lLayer = lMesh->GetLayer(0);
		if (lLayer == NULL)
		{
			lMesh->CreateLayer();
			lLayer = lMesh->GetLayer(0);
		}

		for (std::pair<int, LayerElementUVInfo> _uvInfo : mesh.mUVInfos)
		{
			LayerElementUVInfo uvInfo = _uvInfo.second;
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

		for (int i = 0; i < mesh.mLoopStart.size(); ++i)
		{
			int loopStart = mesh.mLoopStart[i];
			size_t loopEnd = i < mesh.mLoopStart.size() - 1 ? mesh.mLoopStart[i + 1] - 1 : mesh.mIndices.size() - 1;
			lMesh->BeginPolygon(mesh.mMatIndices.size() == 0 ? -1 : mesh.mMatIndices[i]);
			for (int j = loopStart; j <= loopEnd; ++j)
			{
				lMesh->AddPolygon(mesh.mIndices[j]);
			}
			lMesh->EndPolygon();
		}

		FbxGeometryElementSmoothing* lSmoothingElement = nullptr;
		if (mesh.mSmoothMode >= 0)
		{
			lSmoothingElement = lMesh->CreateElementSmoothing();
			lSmoothingElement->SetReferenceMode(FbxLayerElement::eDirect);
		}

		switch (mesh.mSmoothMode)
		{
		case 0:
			lSmoothingElement->SetMappingMode(FbxLayerElement::eByPolygon);
			break;
		case 1:
			lSmoothingElement->SetMappingMode(FbxLayerElement::eByEdge);
			break;
		default:
			mesh.mSmoothing.clear();
		}

		if (lSmoothingElement != nullptr)
		{
			for (int smoothingFlag : mesh.mSmoothing)
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

bool Exporter::BuildPose(FbxScene* pScene, FbxNode*& pMeshNode, FbxNode* pSkeletonNode)
{
	if (pMeshNode == nullptr || pSkeletonNode == nullptr)
	{
		return true;
	}

	FbxPose* lPose = FbxPose::Create(pScene, pMeshNode->GetName());
	lPose->SetIsBindPose(true);

	for (std::pair<std::string, PoseNode> _pose : mPoses)
	{
		const PoseNode& pose = _pose.second;
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

bool Exporter::BuildTakes(FbxScene* pScene, FbxNode* pSkeletonNode)
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

bool Exporter::FillDefaultValueKeys(FbxAnimCurveNode* animCurveNode, FbxAnimCurve* animCurve, const char* channelName, const Channel& channel)
{
	animCurveNode->SetChannelValue(channelName, channel.defaultValue);

	animCurve->KeyModifyBegin();
	FbxTime lTime;
	int lKeyIndex = 0;

	for (const Key& key : channel.keys)
	{
		lTime.SetFramePrecise(key.frame, FbxTime::EMode::eCustom);
		lKeyIndex = animCurve->KeyAdd(lTime);
		animCurve->KeySetValue(lKeyIndex, static_cast<float>(key.value));
		animCurve->KeySetInterpolation(lKeyIndex, FbxAnimCurveDef::eInterpolationCubic);
	}
	animCurve->KeyModifyEnd();

	return true;
}

bool Exporter::CreateScene(FbxScene* pScene)
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

void Exporter::CreateUVInfo(char* meshName, int uvIndex, char* name)
{
	Mesh& mesh = GetMesh(meshName, mMesh);
	GetUVInfo(uvIndex, name, mesh);
}

void Exporter::AddVertex(char* name, double x, double y, double z)
{
	Mesh& mesh = GetMesh(name, mMesh);
	mesh.mVertices.push_back(Vector3(x, y, z));
}

void Exporter::AddNormal(char* name, double x, double y, double z)
{
	Mesh& mesh = GetMesh(name, mMesh);
	mesh.mNormals.push_back(Vector3(x, y, z));
}

void Exporter::AddUV(char* name, int uvIndex, double x, double y)
{
	Mesh& mesh = GetMesh(name, mMesh);
	mesh.mUVInfos.at(uvIndex).mUVs.push_back(UV(x, y));
}

void Exporter::AddUVIndex(char* name, int uvIndex, int index)
{
	Mesh& mesh = GetMesh(name, mMesh);
	mesh.mUVInfos.at(uvIndex).mUVIndices.push_back(index);
}



void Exporter::AddTangent(char* name, Vector3 tangent)
{
	Mesh& mesh = GetMesh(name, mMesh);
	mesh.mTangents.push_back(tangent);
}

void Exporter::AddBinormal(char* name, Vector3 binormal)
{
	Mesh& mesh = GetMesh(name, mMesh);
	mesh.mBinormals.push_back(binormal);
}

void Exporter::SetTangentName(char* name, char* tangentName)
{
	Mesh& mesh = GetMesh(name, mMesh);
	mesh.tangentName = std::string(tangentName);
}

void Exporter::SetBinormalName(char* name, char* binormalName)
{
	Mesh& mesh = GetMesh(name, mMesh);
	mesh.binormalName = std::string(binormalName);
}

void Exporter::AddMatIndex(char* name, int index)
{
	Mesh& mesh = GetMesh(name, mMesh);
	mesh.mMatIndices.push_back(index);
}

void Exporter::AddIndex(char* name, int index)
{
	Mesh& mesh = GetMesh(name, mMesh);
	mesh.mIndices.push_back(index);
}

void Exporter::AddLoopStart(char* name, int start)
{
	Mesh& mesh = GetMesh(name, mMesh);
	mesh.mLoopStart.push_back(start);
}

void Exporter::AddSmoothing(char* name, int smooth)
{
	Mesh& mesh = GetMesh(name, mMesh);
	mesh.mSmoothing.push_back(smooth);
}

void Exporter::SetSmoothMode(char* name, int mode)
{
	Mesh& mesh = GetMesh(name, mMesh);
	mesh.mSmoothMode = mode;
}

void Exporter::AddMeshEdge(char* name, int startVertexIndex, int endVertexIndex)
{
	Mesh& mesh = GetMesh(name, mMesh);
	mesh.edges.push_back(IntVector2(startVertexIndex, endVertexIndex));
}

void Exporter::SetMeshProperty(char* name, Vector3 trans, Vector3 rot, Vector3 sca)
{
	Mesh& mesh = GetMesh(name, mMesh);
	mesh.mMeshName = std::string(name);
	mesh.lclTranslation = trans;
	mesh.lclRotation = rot;
	mesh.lclScaling = sca;
}

void Exporter::AddMaterial(char* mName, char* sName, Vector3 diffuse, Vector3 ambient, Vector3 emissive)
{
	Material mat = Material(mName, sName, diffuse, ambient, emissive);
	mMaterials.push_back(mat);
}

void Exporter::AddBone(char* name, Vector3 lclTranslation, Vector3 lclRotation, Vector3 lclScaling)
{
	mBones.insert(make_pair(std::string(name), Bone(name, lclTranslation, lclRotation, lclScaling)));
}

void Exporter::AddBoneChild(char* child, char* parent)
{
	Bone& parentB = mBones.at(std::string(parent));
	Bone& childB = mBones.at(std::string(child));
	childB.parentName = parentB.boneName;
}

void Exporter::SetTextureMatProp(char* name, char* matName, char* matProp)
{
	Texture& tex = mTextures.at(std::string(name));
	tex.parentMat.push_back(matName);
	tex.matProp = std::string(matProp);
}

void Exporter::AddTexture(char* name, char* fileName, char* relFileName, int alphaSource, bool premultiplyAlpha, int currentMappingType, char* UVSet, int wrapModeU, int wrapModeV,
	Vector3 translation, Vector3 scaling, bool useMaterial, bool useMipMap)
{
	mTextures.insert(make_pair(std::string(name), Texture(name, fileName, relFileName, alphaSource, premultiplyAlpha, currentMappingType, UVSet, wrapModeU, wrapModeV,
		translation, scaling, useMaterial, useMipMap)));
}

void Exporter::AddPoseNode(char* name, Mat4x4 transform)
{
	mPoses.insert(make_pair(std::string(name), PoseNode(name, transform)));
}

SubDeformer& Exporter::GetSubDeformer(const char* mName, const char* bName)
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

void Exporter::SetSubDeformerTransformLink(char* mName, char* bName, Mat4x4 transformLink)
{
	SubDeformer& subDeformer = GetSubDeformer(mName, bName);
	subDeformer.transformLink = transformLink;
}

void Exporter::SetSubDeformerTransform(char* mName, char* bName, Mat4x4 transform, Vector4 quat)
{
	SubDeformer& subDeformer = GetSubDeformer(mName, bName);
	subDeformer.transform = transform;
	subDeformer.quat = quat;
}

void Exporter::AddSubDeformerWeight(char* mName, char* bName, double weight)
{
	SubDeformer& subDeformer = GetSubDeformer(mName, bName);
	subDeformer.weights.push_back(weight);
}

void Exporter::AddSubDeformerIndex(char* mName, char* bName, int index)
{
	SubDeformer& subDeformer = GetSubDeformer(mName, bName);
	subDeformer.indexes.push_back(index);
}

ModelAnim& Exporter::GetModelAnim(char* tName, char* mName)
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

void Exporter::SetTimeSpan(char* tName, double lStart, double lEnd, double rStart, double rEnd)
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

void Exporter::SetChannelDefaultValue(char* takeName, char* modelName, int type, double value)
{
	if (type >= ChannelType::ChannelMax)
	{
		return;
	}

	ChannelType channelType = static_cast<ChannelType>(type);
	ModelAnim& model = GetModelAnim(takeName, modelName);
	
	model.channels[channelType].defaultValue = value;
}

void Exporter::AddChannelKey(char* takeName, char* modelName, int type, double frame, double value)
{
	if (type >= ChannelType::ChannelMax)
	{
		return;
	}

	ChannelType channelType = static_cast<ChannelType>(type);
	ModelAnim& model = GetModelAnim(takeName, modelName);

	model.channels[channelType].keys.push_back(Key(frame, value));
}

void Exporter::SetFPS(double fps)
{
	mFps = fps;
}

void Exporter::PrintTakes()
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
			PrintModelAnim(_model.second);
		}
	}
}

void Exporter::PrintSkeleton()
{
	for (std::pair<std::string, Bone> _bone : mBones)
	{
		PrintBone(_bone.second);
	}

	for (std::pair<std::string, Deformer> _deformer : mDeformers)
	{
		const Deformer& deformer = _deformer.second;
		std::cout << "deformer name: " << deformer.deformerName << std::endl;
		for (std::pair<std::string, SubDeformer> _subDeformer : deformer.subDeformers)
		{
			PrintSubDeformer(_subDeformer.second);
		}
	}

	for (std::pair<std::string, PoseNode> _pose : mPoses)
	{
		PrintPoseNode(_pose.second);
	}
}

void Exporter::PrintMeshProps(const Mesh& mesh)
{
	std::cout << "mesh translation: " << mesh.lclTranslation << " rotation: " << mesh.lclRotation << " scale: " << mesh.lclScaling << std::endl;
	std::cout << "mesh GeometricTranslation: " << mesh.GeometricTranslation << " GeometricRotation: " << mesh.GeometricRotation << " GeometricScaling: " << mesh.GeometricScaling << std::endl;
	std::cout << "mesh RotationOffset: " << mesh.RotationOffset << " RotationPivot: " << mesh.RotationPivot << " ScalingOffset: " << mesh.ScalingOffset << std::endl;
	std::cout << "mesh ScalingPivot: " << mesh.ScalingPivot << " PreRotation: " << mesh.PreRotation << " PostRotation: " << mesh.PostRotation << std::endl;
	std::cout << "mesh RotationOrder: " << mesh.RotationOrder << " RotationActive: " << mesh.RotationActive << std::endl;
}

void Exporter::PrintMesh()
{
	for (std::pair<std::string, Mesh> _mesh : mMesh)
	{
		FBXUtil::PrintMesh(_mesh.second);
		PrintMeshProps(_mesh.second);
	}

	for (const Material& mat : mMaterials)
	{
		PrintMaterial(mat);
	}

	std::cout << "Textures:" << std::endl;
	for (std::pair<std::string, Texture> _tex : mTextures)
	{
		PrintTexture(_tex.second);
	}
}

bool Exporter::Export(char* filePath)
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



extern "C"
{
	DLLEXPORT Exporter* Exporter_New() { return new Exporter(); }
	DLLEXPORT void Exporter_AddVertex(Exporter* exporter, char* name, double x, double y, double z) { exporter->AddVertex(name, x, y, z); }
	DLLEXPORT void Exporter_AddNormal(Exporter* exporter, char* name, double x, double y, double z) { exporter->AddNormal(name, x, y, z); }
	DLLEXPORT void Exporter_CreateUVInfo(Exporter* exporter, char* meshName, int uvIndex, char* name) { exporter->CreateUVInfo(meshName, uvIndex, name); }
	DLLEXPORT void Exporter_AddUV(Exporter* exporter, char* name, int uvIndex, double x, double y) { exporter->AddUV(name, uvIndex, x, y); }
	DLLEXPORT void Exporter_AddIndex(Exporter* exporter, char* name, int n) { exporter->AddIndex(name, n); }
	DLLEXPORT void Exporter_AddMatIndex(Exporter* exporter, char* name, int n) { exporter->AddMatIndex(name, n); }
	DLLEXPORT void Exporter_AddTangent(Exporter* exporter, char* name, FBXUtil::Vector3 tan) { exporter->AddTangent(name, tan); }
	DLLEXPORT void Exporter_AddBinormal(Exporter* exporter, char* name, FBXUtil::Vector3 bino) { exporter->AddBinormal(name, bino); }
	DLLEXPORT void Exporter_SetTangentName(Exporter* exporter, char* name, char* tangentName) { exporter->SetTangentName(name, tangentName); }
	DLLEXPORT void Exporter_SetBinormalName(Exporter* exporter, char* name, char* binormalName) { exporter->SetBinormalName(name, binormalName); }
	DLLEXPORT void Exporter_AddUVIndex(Exporter* exporter, char* name, int uvIndex, int n) { exporter->AddUVIndex(name, uvIndex, n); }
	DLLEXPORT void Exporter_AddLoopStart(Exporter* exporter, char* name, int n) { exporter->AddLoopStart(name, n); }
	DLLEXPORT void Exporter_AddMeshEdge(Exporter* exporter, char* name, int s, int e) { exporter->AddMeshEdge(name, s, e); }
	DLLEXPORT void Exporter_AddSmoothing(Exporter* exporter, char* name, int s) { exporter->AddSmoothing(name, s); }
	DLLEXPORT void Exporter_SetSmoothMode(Exporter* exporter, char* name, int m) { exporter->SetSmoothMode(name, m); }
	DLLEXPORT void Exporter_SetMeshProperty(Exporter* exporter, char* name, FBXUtil::Vector3 trans, FBXUtil::Vector3 rot, FBXUtil::Vector3 sca)
		{ exporter->SetMeshProperty(name, trans, rot, sca); }
	DLLEXPORT void Exporter_AddMaterial(Exporter* exporter, char* mName, char* sName, FBXUtil::Vector3 diffuse, FBXUtil::Vector3 ambient, FBXUtil::Vector3 emissive)
	{ exporter->AddMaterial(mName, sName, diffuse, ambient, emissive); }
	DLLEXPORT void Exporter_AddTexture(Exporter* exporter, char* name, char* fileName, char* relFileName, int alphaSource, bool premultiplyAlpha, int currentMappingType, char* UVSet, int wrapModeU, int wrapModeV,
		FBXUtil::Vector3 translation, FBXUtil::Vector3 scaling, bool useMaterial, bool useMipMap) { exporter->AddTexture(name, fileName, relFileName, alphaSource, premultiplyAlpha, currentMappingType, UVSet,
			wrapModeU, wrapModeV, translation, scaling, useMaterial, useMipMap);}
	DLLEXPORT void Exporter_SetTextureMatProp(Exporter* exporter, char* name, char* matName, char* matProp) { exporter->SetTextureMatProp(name, matName, matProp); }

	DLLEXPORT void Exporter_AddPoseNode(Exporter* exporter, char* name, FBXUtil::Mat4x4 mat) { exporter->AddPoseNode(name, mat); }
	DLLEXPORT void Exporter_AddBoneChild(Exporter* exporter, char* cName, char* pName) { exporter->AddBoneChild(cName, pName); }
	DLLEXPORT void Exporter_AddBone(Exporter* exporter, char* name, FBXUtil::Vector3 lclTranslation, FBXUtil::Vector3 lclRotation, FBXUtil::Vector3 lclScaling)
		{ exporter->AddBone(name, lclTranslation, lclRotation, lclScaling); }
	DLLEXPORT void Exporter_AddSubDeformerIndex(Exporter* exporter, char* mName, char* bName, int index) { exporter->AddSubDeformerIndex(mName, bName, index); }
	DLLEXPORT void Exporter_AddSubDeformerWeight(Exporter* exporter, char* mName, char* bName, double weight) { exporter->AddSubDeformerWeight(mName, bName, weight); }
	DLLEXPORT void Exporter_SetSubDeformerTransform(Exporter* exporter, char* mName, char* bName, FBXUtil::Mat4x4 transf, FBXUtil::Vector4 quat) { exporter->SetSubDeformerTransform(mName, bName, transf, quat); }
	DLLEXPORT void Exporter_SetSubDeformerTransformLink(Exporter* exporter, char* mName, char* bName, FBXUtil::Mat4x4 transfLink) { exporter->SetSubDeformerTransformLink(mName, bName, transfLink); }

	DLLEXPORT void Exporter_SetFPS(Exporter* exporter, double fps) { exporter->SetFPS(fps); }
	DLLEXPORT void Exporter_SetTimeSpan(Exporter* exporter, char* tName, double lStart, double lEnd, double rStart, double rEnd) { exporter->SetTimeSpan(tName, lStart, lEnd, rStart, rEnd); }
	DLLEXPORT void Exporter_SetChannelDefaultValue(Exporter* exporter, char* takeName, char* modelName, int type, double value) { exporter->SetChannelDefaultValue(takeName, modelName, type, value); }
	DLLEXPORT void Exporter_AddChannelKey(Exporter* exporter, char* takeName, char* modelName, int type, double frame, double value) { exporter->AddChannelKey(takeName, modelName, type, frame, value); }

	DLLEXPORT bool Exporter_Export(Exporter* exporter, char* name) { return exporter->Export(name); }
	DLLEXPORT void Exporter_PrintMesh(Exporter* exporter) { exporter->PrintMesh(); }
	DLLEXPORT void Exporter_PrintSkeleton(Exporter* exporter) { exporter->PrintSkeleton(); }
	DLLEXPORT void Exporter_PrintTakes(Exporter* exporter) { exporter->PrintTakes(); }

	DLLEXPORT void Exporter_SetAsASCII(Exporter* exporter, bool asAscii) { exporter->SetAsASCII(asAscii); }
}
