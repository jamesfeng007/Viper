#include "Importer.h"
#include "Common/Common.h"

Importer::Importer()
{
	mLogFile = std::ofstream("ImportFBXSdk.log");
	mCoutbuf = std::cout.rdbuf(); //save old buf
	std::cout.rdbuf(mLogFile.rdbuf()); //redirect std::cout to out.txt!

	mMesh.clear();
	mModels.clear();
	mMaterials.clear();
	mTextures.clear();
	mConnections.clear();
	mBones.clear();
	mPoses.clear();
	mSubDeformers.clear();
	mDeformers.clear();
	mAnims.clear();
	mLayers.clear();
	mStacks.clear();
}

Importer::~Importer()
{
	std::cout.rdbuf(mCoutbuf); //reset to standard output again
	mLogFile.flush();
	mLogFile.close();
}

bool Importer::Import(char* filePath)
{
	FbxManager* lSdkManager = NULL;
	FbxScene* lScene = NULL;
	bool lResult;

	InitializeSdkObjects(lSdkManager, lScene);
	FbxString lFilePath(filePath);

	if (lFilePath.IsEmpty())
	{
		lResult = false;
		PrintString("\n\nUsage: ImportScene <FBX file name>\n\n");
	}
	else
	{
		PrintString("\n\nFile: %s\n\n", lFilePath.Buffer());
		lResult = LoadScene(lSdkManager, lScene, lFilePath.Buffer());
	}

	if (lResult == false)
	{
		PrintString("\n\nAn error occurred while loading the scene...");
	}
	else
	{
		AnalyzeGlobalSettings(&lScene->GetGlobalSettings());
		AnalyzeContent(lScene);
		AnalyzePose(lScene);
		AnalyzeAnimation(lScene);
	}


	// Destroy all objects created by the FBX SDK.
	DestroySdkObjects(lSdkManager, lResult);

	return lResult;
}

bool Importer::GetGlobalSettings(GlobalSettings* globalSettings)
{
	if (globalSettings == nullptr)
	{
		return false;
	}

	globalSettings->UnitScaleFactor = mGlobalSettings.UnitScaleFactor;
	globalSettings->OriginalUnitScaleFactor = mGlobalSettings.OriginalUnitScaleFactor;
	globalSettings->CustomFrameRate = mGlobalSettings.CustomFrameRate;
	globalSettings->TimeMode = mGlobalSettings.TimeMode;
	strcpy_s(globalSettings->AxisForward, 3, mGlobalSettings.AxisForward);
	strcpy_s(globalSettings->AxisUp, 3, mGlobalSettings.AxisUp);

	return true;
}

FbxUInt64 Importer::GetModelUUID(int index)
{
	Node& model = mModels.at(index);
	return model.uuid;
}

FbxUInt64 Importer::GetMeshUUID(int index)
{
	Mesh& mesh = mMesh.at(index);
	return mesh.uuid;
}

bool Importer::GetModelTransformProp(int index, ObjectTransformProp* prop)
{
	Node& node = mModels.at(index);
	if (prop == nullptr)
	{
		return false;
	}

	prop->lclTranslation = node.lclTranslation;
	prop->lclRotation = node.lclRotation;
	prop->lclScaling = node.lclScaling;
	prop->GeometricTranslation = node.GeometricTranslation;
	prop->GeometricRotation = node.GeometricRotation;
	prop->GeometricScaling = node.GeometricScaling;
	prop->RotationOffset = node.RotationOffset;
	prop->RotationPivot = node.RotationPivot;
	prop->ScalingOffset = node.ScalingOffset;
	prop->ScalingPivot = node.ScalingPivot;
	prop->PreRotation = node.PreRotation;
	prop->PostRotation = node.PostRotation;
	prop->RotationOrder = node.RotationOrder;
	prop->RotationActive = node.RotationActive;

	return true;
}

int Importer::GetMeshVerticeSize(int index)
{
	Mesh& mesh = mMesh.at(index);
	return static_cast<int>(mesh.mVertices.size());
}

int Importer::GetMeshIndiceSize(int index)
{
	Mesh& mesh = mMesh.at(index);
	return static_cast<int>(mesh.mIndices.size());
}

int Importer::GetMeshNormalSize(int index)
{
	Mesh& mesh = mMesh.at(index);
	return static_cast<int>(mesh.mNormals.size());
}

int Importer::GetMeshSmoothingSize(int index)
{
	Mesh& mesh = mMesh.at(index);
	return static_cast<int>(mesh.mSmoothing.size());
}

int Importer::GetMeshUVInfoSize(int index)
{
	Mesh& mesh = mMesh.at(index);
	return static_cast<int>(mesh.mUVInfos.size());
}

int Importer::GetMeshUVIndiceSize(int index, int uvIndex)
{
	LayerElementUVInfo& uvInfo = mMesh.at(index).mUVInfos.at(uvIndex);
	return static_cast<int>(uvInfo.mUVIndices.size());
}

int Importer::GetMeshUVVerticeSize(int index, int uvIndex)
{
	LayerElementUVInfo& uvInfo = mMesh.at(index).mUVInfos.at(uvIndex);
	return static_cast<int>(uvInfo.mUVs.size());
}

int Importer::GetMeshMatIndiceSize(int index)
{
	return static_cast<int>(mMesh.at(index).mMatIndices.size());
}

bool Importer::GetMeshMaterialInfo(int index, int* pMatIndex, long indiceSize, LayerElementInfo* layerElemInfo)
{
	if (pMatIndex == nullptr)
	{
		return false;
	}

	Mesh& mesh = mMesh.at(index);
	std::vector<int>& indices = mesh.mMatIndices;
	if (indiceSize < indices.size())
	{
		return false;
	}

	for (int i = 0; i < indices.size(); ++i)
	{
		pMatIndex[i] = indices.at(i);
	}

	if (mesh.mMatMapping == 0)
	{
		sprintf_s(layerElemInfo->MappingType, 32, "%s", "ByPolygon");
	}
	else if (mesh.mMatMapping == 1)
	{
		sprintf_s(layerElemInfo->MappingType, 32, "%s", "AllSame");
	}
	
	if (mesh.mMatRef == 2)
	{
		sprintf_s(layerElemInfo->RefType, 32, "%s", "IndexToDirect");
	}
	
	return true;
}

bool Importer::GetMeshUVIndice(int index, int uvIndex, int* pIndice, long indiceSize)
{
	if (pIndice == nullptr)
	{
		return false;
	}

	LayerElementUVInfo& uvInfo = mMesh.at(index).mUVInfos.at(uvIndex);
	std::vector<int>& indices = uvInfo.mUVIndices;
	if (indiceSize < indices.size())
	{
		return false;
	}

	for (int i = 0; i < indices.size(); ++i)
	{
		pIndice[i] = indices.at(i);
	}

	return true;
}

bool Importer::GetMeshUVVertice(int index, int uvIndex, double* pVertice, long verticeSize)
{
	if (pVertice == nullptr)
	{
		return false;
	}

	LayerElementUVInfo& uvInfo = mMesh.at(index).mUVInfos.at(uvIndex);
	std::vector<UV>& vertices = uvInfo.mUVs;
	if (verticeSize < vertices.size() * 2)
	{
		return false;
	}

	for (int i = 0; i < vertices.size(); ++i)
	{
		pVertice[2 * i] = vertices.at(i).x;
		pVertice[2 * i + 1] = vertices.at(i).y;
	}

	return true;
}

const char* Importer::GetUVInfoName(int index, int uvIndex, LayerElementInfo* layerElemInfo)
{
	Mesh& mesh = mMesh.at(index);

	sprintf_s(layerElemInfo->MappingType, 32, "%s", "ByPolygonVertex");
	sprintf_s(layerElemInfo->RefType, 32, "%s", "IndexToDirect");

	return mesh.mUVInfos.at(uvIndex).name.c_str();
}

int Importer::GetMeshEdgeSize(int index)
{
	Mesh& mesh = mMesh.at(index);
	return static_cast<int>(mesh.edges.size());
}

bool Importer::GetMeshSmoothings(int index, int* pSmoothings, long smoothingSize, LayerElementInfo* layerElemInfo)
{
	if (pSmoothings == nullptr || layerElemInfo == nullptr)
	{
		return false;
	}

	std::vector<int>& smoothings = mMesh.at(index).mSmoothing;
	if (smoothingSize < smoothings.size())
	{
		return false;
	}

	for (int i = 0; i < smoothings.size(); ++i)
	{
		pSmoothings[i] = smoothings.at(i);
	}

	if (mMesh.at(index).mSmoothMode == 0)
	{
		sprintf_s(layerElemInfo->MappingType, 32, "%s", "ByPolygon");
	}
	else if (mMesh.at(index).mSmoothMode == 1)
	{
		sprintf_s(layerElemInfo->MappingType, 32, "%s", "ByEdge");
	}
	
	sprintf_s(layerElemInfo->RefType, 32, "%s", "Direct");

	return true;
}

bool Importer::GetMeshNormals(int index, double* pNormals, long normalSize, LayerElementInfo *layerElemInfo)
{
	if (pNormals == nullptr || layerElemInfo == nullptr)
	{
		return false;
	}

	std::vector<Vector3>& normals = mMesh.at(index).mNormals;
	if (normalSize < normals.size() * 3)
	{
		return false;
	}

	for (int i = 0; i < normals.size(); ++i)
	{
		pNormals[3 * i] = normals.at(i).x;
		pNormals[3 * i + 1] = normals.at(i).y;
		pNormals[3 * i + 2] = normals.at(i).z;
	}

	sprintf_s(layerElemInfo->MappingType, 32, "%s", mMesh.at(index).normalMapType.c_str());
	sprintf_s(layerElemInfo->RefType, 32, "%s", mMesh.at(index).normalRefType.c_str());

	return true;
}

bool Importer::GetConnections(UInt64Vector2* pConnection, long connectionSize)
{
	if (pConnection == nullptr)
	{
		return false;
	}

	if (connectionSize < mConnections.size())
	{
		return false;
	}

	for (int i = 0; i < mConnections.size(); ++i)
	{
		pConnection[i].x = mConnections.at(i).x;
		pConnection[i].y = mConnections.at(i).y;
	}

	return true;
}

bool Importer::GetMeshVertice(int index, double* pVertice, long verticeSize)
{
	if (pVertice == nullptr)
	{
		return false;
	}

	std::vector<Vector3>& vertices = mMesh.at(index).mVertices;
	if (verticeSize < vertices.size() * 3)
	{
		return false;
	}

	for (int i = 0; i < vertices.size(); ++i)
	{
		pVertice[3*i] = vertices.at(i).x;
		pVertice[3*i+1] = vertices.at(i).y;
		pVertice[3*i+2] = vertices.at(i).z;
	}
	
	return true;
}

bool Importer::GetMeshEdges(int index, int* pEdges, long edgeSize)
{
	if (pEdges == nullptr)
	{
		return false;
	}

	std::vector<IntVector2>& edges = mMesh.at(index).edges;
	if (edgeSize < edges.size())
	{
		return false;
	}

	for (int i = 0; i < edges.size(); ++i)
	{
		pEdges[2 * i] = edges.at(i).x;
		pEdges[2 * i + 1] = edges.at(i).y;
	}

	return true;
}

bool Importer::GetMeshIndice(int index, int* pIndice, long indiceSize)
{
	if (pIndice == nullptr)
	{
		return false;
	}

	std::vector<int>& indices = mMesh.at(index).mIndices;
	if (indiceSize < indices.size())
	{
		return false;
	}

	for (int i = 0; i < indices.size(); ++i)
	{
		pIndice[i] = indices.at(i);
	}

	return true;
}

const char* Importer::GetMaterialName(int index)
{
	Material& material = mMaterials.at(index);
	return material.materialName.c_str();
}

bool Importer::GetMaterialProps(int index, Vector3* pEmissive, Vector3* pAmbient, Vector3* pDiffuse, MatProps* pExtra)
{
	if (pEmissive == nullptr || pAmbient == nullptr || pDiffuse == nullptr || pExtra == nullptr)
	{
		return false;
	}

	Material& material = mMaterials.at(index);
	pEmissive->x = material.emissiveColor[0];
	pEmissive->y = material.emissiveColor[1];
	pEmissive->z = material.emissiveColor[2];

	pAmbient->x = material.ambientColor[0];
	pAmbient->y = material.ambientColor[1];
	pAmbient->z = material.ambientColor[2];

	pDiffuse->x = material.diffuseColor[0];
	pDiffuse->y = material.diffuseColor[1];
	pDiffuse->z = material.diffuseColor[2];

	pExtra->BumpFactor = material.extra.BumpFactor;

	return true;
}

FbxUInt64 Importer::GetRefBoneUUID(int index)
{
	PoseNode& node = mPoses.at(index);
	return node.refNodeUuid;
}

bool Importer::GetClusterTransforms(int index, double* pTransform, double* pLinkTransform, int matSize)
{
	if (pTransform == nullptr || pLinkTransform == nullptr)
	{
		return false;
	}

	Mat4x4& mat = mSubDeformers.at(index).transform;
	if (matSize < mat.Size())
	{
		return false;
	}

	Mat4x4* pMat = reinterpret_cast<Mat4x4*>(pTransform);

	pMat->Fill(mat);

	mat = mSubDeformers.at(index).transformLink;
	if (matSize < mat.Size())
	{
		return false;
	}

	pMat = reinterpret_cast<Mat4x4*>(pLinkTransform);

	pMat->Fill(mat);

	return true;
}

bool Importer::GetPoseMatrix(int index, double* pV, int matSize)
{
	if (pV == nullptr)
	{
		return false;
	}

	Mat4x4& mat = mPoses.at(index).poseTransform;
	if (matSize < mat.Size())
	{
		return false;
	}

	Mat4x4* pMat = reinterpret_cast<Mat4x4*>(pV);

	pMat->Fill(mat);

	return true;
}

bool Importer::GetClusterWeightIndice(int index, int* pIndice, double* pWeight, long indiceSize)
{
	if (pIndice == nullptr || pWeight == nullptr)
	{
		return false;
	}

	std::vector<int>& indices = mSubDeformers.at(index).indexes;
	std::vector<double> weights = mSubDeformers.at(index).weights;
	if (indiceSize < indices.size() || weights.size() != indices.size())
	{
		return false;
	}

	for (int i = 0; i < indices.size(); ++i)
	{
		pIndice[i] = indices.at(i);
		pWeight[i] = weights.at(i);
	}

	return true;
}

int Importer::GetClusterIndiceSize(int index)
{
	SubDeformer& cluster = mSubDeformers.at(index);
	return static_cast<int>(cluster.indexes.size());
}

const char* Importer::GetStackName(int index)
{
	return mStacks.at(index).refName.c_str();
}

const char* Importer::GetLayerName(FbxUInt64 uuid)
{
	std::vector<NameUUID>::iterator ite = std::find_if(mLayers.begin(), mLayers.end(), [uuid](const NameUUID& item) { return item.refUUID == uuid; });
	if (ite != std::end(mLayers))
	{
		return (*ite).refName.c_str();
	}

	return "";
}

double Importer::GetAnimChannelDefaultValue(FbxUInt64 stackUUID, FbxUInt64 layerUUID, FbxUInt64 boneUUID, ChannelType channel)
{
	double value = 0.0;
	std::vector<ModelAnim>::iterator ite = std::find_if(mAnims.begin(), mAnims.end(), [stackUUID, layerUUID, boneUUID](const ModelAnim& anim) 
	{ return anim.refStackUUID == stackUUID && anim.refLayerUUID == layerUUID && anim.refModelUUID == boneUUID; });
	if (ite != std::end(mAnims) && channel < ChannelMax && channel >= T_X)
	{
		value = (*ite).channels[channel].defaultValue;
	}

	return value;
}

bool Importer::GetKeyTimeValue(FbxUInt64 stackUUID, FbxUInt64 layerUUID, FbxUInt64 boneUUID, ChannelType channel, FbxLongLong* pTimes, double* pValues, int keyCount)
{
	if (pTimes == nullptr || pValues == nullptr)
	{
		return false;
	}

	std::vector<Key> keys;
	std::vector<ModelAnim>::iterator ite = std::find_if(mAnims.begin(), mAnims.end(), [stackUUID, layerUUID, boneUUID](const ModelAnim& anim)
	{ return anim.refStackUUID == stackUUID && anim.refLayerUUID == layerUUID && anim.refModelUUID == boneUUID; });
	if (ite != std::end(mAnims) && channel < ChannelMax && channel >= T_X)
	{
		keys = (*ite).channels[channel].keys;
	}

	if (keys.size() == 0)
	{
		return false;
	}

	if (keyCount < keys.size())
	{
		return false;
	}

	for (int i = 0; i < keys.size(); ++i)
	{
		pTimes[i] = keys.at(i).timeValue;
		pValues[i] = keys.at(i).value;
	}

	return true;
}

int Importer::GetKeyCount(FbxUInt64 stackUUID, FbxUInt64 layerUUID, FbxUInt64 boneUUID, ChannelType channel)
{
	std::vector<ModelAnim>::iterator ite = std::find_if(mAnims.begin(), mAnims.end(), [stackUUID, layerUUID, boneUUID](const ModelAnim& anim)
	{ return anim.refStackUUID == stackUUID && anim.refLayerUUID == layerUUID && anim.refModelUUID == boneUUID; });
	if (ite != std::end(mAnims) && channel < ChannelMax && channel >= T_X)
	{
		return static_cast<int>((*ite).channels[channel].keys.size());
	}

	return 0;
}

bool Importer::AnimationExist(FbxUInt64 stackUUID, FbxUInt64 layerUUID, FbxUInt64 boneUUID)
{
	std::vector<ModelAnim>::iterator ite = std::find_if(mAnims.begin(), mAnims.end(), [stackUUID, layerUUID, boneUUID](const ModelAnim& anim)
	{ return anim.refStackUUID == stackUUID && anim.refLayerUUID == layerUUID && anim.refModelUUID == boneUUID; });

	if (ite != std::end(mAnims))
	{
		for (int i = T_X; i < ChannelMax; ++i)
		{
			if ((*ite).channels[i].keys.size() > 0)
			{
				return true;
			}
		}
	}

	return false;
}

FbxUInt64 Importer::GetStackUUID(int index)
{
	return mStacks.at(index).refUUID;
}

FbxUInt64 Importer::GetSkinUUID(int index)
{
	Deformer& skin = mDeformers.at(index);
	return skin.uuid;
}

FbxUInt64 Importer::GetClusterUUID(int index)
{
	SubDeformer& cluster = mSubDeformers.at(index);
	return cluster.uuid;
}

FbxUInt64 Importer::GetBoneUUID(int index)
{
	Bone& bone = mBones.at(index);
	return bone.uuid;
}

FbxUInt64 Importer::GetTextureUUID(int index)
{
	Texture& tex = mTextures.at(index);
	return tex.uuid;
}

const char* Importer::GetTextureRelFileName(int index)
{
	Texture& tex = mTextures.at(index);
	return tex.relFileName.c_str();
}

const char* Importer::GetTextureMatProp(int index)
{
	Texture& tex = mTextures.at(index);
	return tex.matProp.c_str();
}

bool Importer::GetTextureMapping(int index, Vector3* pTranslation, Vector3* pRotation, Vector3* pScaling, IntVector2* pWrapMode)
{
	if (pTranslation == nullptr || pRotation == nullptr || pScaling == nullptr || pWrapMode == nullptr)
	{
		return false;
	}

	Texture& tex = mTextures.at(index);
	pTranslation->Fill(tex.translation);
	pRotation->Fill(tex.rotation);
	pScaling->Fill(tex.scaling);
	pWrapMode->Fill(IntVector2(tex.wrapModeU, tex.wrapModeV));

	return true;
}

const char* Importer::GetTextureFileName(int index)
{
	Texture& tex = mTextures.at(index);
	return tex.fileName.c_str();
}

const char* Importer::GetTextureName(int index)
{
	Texture& tex = mTextures.at(index);
	return tex.name.c_str();
}

const char* Importer::GetBoneName(int index)
{
	Bone& bone = mBones.at(index);
	return bone.boneName.c_str();
}

const char* Importer::GetSkinName(int index)
{
	Deformer& skin = mDeformers.at(index);
	return skin.deformerName.c_str();
}

const char* Importer::GetClusterName(int index)
{
	SubDeformer& cluster = mSubDeformers.at(index);
	return cluster.subDeformerName.c_str();
}

FbxUInt64 Importer::GetMaterialUUID(int index)
{
	Material& material = mMaterials.at(index);
	return material.uuid;
}

Texture& Importer::GetTexture(FbxUInt64 uuid, bool exist)
{
	std::vector<Texture>::iterator ite = std::find_if(mTextures.begin(), mTextures.end(), [uuid](const Texture& tex) { return tex.uuid == uuid; });
	exist = true;
	if (ite == std::end(mTextures))
	{
		Texture tex;
		tex.uuid = uuid;
		mTextures.push_back(tex);
		exist = false;
		return mTextures.back();
	}
	
	return *ite;
}

void Importer::AnalyzeTexture(FbxProperty& prop, FbxSurfaceMaterial* lMaterial)
{
	if (prop.IsValid())
	{
		int lTextureCount = prop.GetSrcObjectCount<FbxTexture>();
		for (int j = 0; j < lTextureCount; ++j)
		{
			FbxTexture* lTexture = prop.GetSrcObject<FbxTexture>(j);
			if (lTexture)
			{
				mConnections.push_back(UInt64Vector2(lMaterial->GetUniqueID(), lTexture->GetUniqueID()));
				bool exist = false;
				Texture& tex = GetTexture(lTexture->GetUniqueID(), exist);
				tex.parentMat.push_back(std::string(lMaterial->GetName()));
				if (!exist)
				{
					tex.name = lTexture->GetName();
					tex.scaling = Vector3(lTexture->GetScaleU(), lTexture->GetScaleV(), 0.0);
					tex.translation = Vector3(lTexture->GetTranslationU(), lTexture->GetTranslationV(), 0.0);
					tex.rotation = Vector3(lTexture->GetRotationU(), lTexture->GetRotationV(), lTexture->GetRotationW());
					tex.alphaSource = lTexture->GetAlphaSource();
					tex.currentMappingType = lTexture->GetMappingType();
					tex.premultiplyAlpha = lTexture->GetPremultiplyAlpha();
					tex.UVSet = std::string(lTexture->UVSet.Get());
					tex.wrapModeU = lTexture->GetWrapModeU();
					tex.wrapModeV = lTexture->GetWrapModeV();
					tex.matProp = prop.GetName();
					FbxFileTexture *lFileTexture = FbxCast<FbxFileTexture>(lTexture);
					if (lFileTexture)
					{
						tex.fileName = std::string(lFileTexture->GetFileName());
						tex.relFileName = std::string(lFileTexture->GetRelativeFileName());
						tex.useMaterial = lFileTexture->UseMaterial.Get();
						tex.useMipMap = lFileTexture->UseMipMap.Get();
					}
				}
			}
		}
	}
}

void Importer::AnalyzeMaterial(FbxNode* pNode)
{
	int lMaterialCount = pNode->GetMaterialCount();
	FbxPropertyT<FbxDouble3> lKFbxDouble3;
	FbxPropertyT<FbxString> lString;
	FbxProperty lProperty;

	for (int i = 0; i < lMaterialCount; ++i)
	{
		FbxSurfaceMaterial* lMaterial = pNode->GetMaterial(i);
		if (lMaterial != nullptr)
		{
			mConnections.push_back(UInt64Vector2(pNode->GetUniqueID(), lMaterial->GetUniqueID()));
			if (lMaterial->GetClassId().Is(FbxSurfacePhong::ClassId))
			{
				Material material = Material();
				material.uuid = lMaterial->GetUniqueID();
				lKFbxDouble3 = ((FbxSurfacePhong *)lMaterial)->Ambient;
				material.ambientColor = lKFbxDouble3.Get();
				lKFbxDouble3 = ((FbxSurfacePhong *)lMaterial)->Diffuse;
				material.diffuseColor = lKFbxDouble3.Get();
				lKFbxDouble3 = ((FbxSurfacePhong *)lMaterial)->Emissive;
				material.emissiveColor = lKFbxDouble3.Get();
				lString = lMaterial->ShadingModel;
				material.shadingName = std::string(lString.Get().Buffer());
				material.materialName = std::string(lMaterial->GetName());
				material.extra.BumpFactor = ((FbxSurfaceLambert*)lMaterial)->BumpFactor.Get();
				mMaterials.push_back(material);
			}
			
			int lTextureIndex;
			FBXSDK_FOR_EACH_TEXTURE(lTextureIndex)
			{
				lProperty = lMaterial->FindProperty(FbxLayerElement::sTextureChannelNames[lTextureIndex]);
				AnalyzeTexture(lProperty, lMaterial);
			}
		}
	}
}

void Importer::AnalyzeLink(FbxGeometry* pGeometry)
{
	int lSkinCount = pGeometry->GetDeformerCount(FbxDeformer::eSkin);
	for (int i = 0; i != lSkinCount; ++i)
	{
		FbxSkin* lSkin = (FbxSkin*)(pGeometry->GetDeformer(i, FbxDeformer::eSkin));
		if (lSkin != nullptr)
		{
			Deformer deformer;
			deformer.deformerName = lSkin->GetName();
			deformer.uuid = lSkin->GetUniqueID();
			mDeformers.push_back(deformer);
			mConnections.push_back(UInt64Vector2(pGeometry->GetUniqueID(), lSkin->GetUniqueID()));
			int lClusterCount = lSkin->GetClusterCount();
			for (int j = 0; j != lClusterCount; ++j)
			{
				FbxCluster* lCluster = lSkin->GetCluster(j);
				SubDeformer subDeformer;
				if (lCluster != nullptr)
				{
					if (lCluster->GetLinkMode() == FbxCluster::eTotalOne)
					{
						mConnections.push_back(UInt64Vector2(lSkin->GetUniqueID(), lCluster->GetUniqueID()));
						int lIndexCount = lCluster->GetControlPointIndicesCount();
						int* lIndices = lCluster->GetControlPointIndices();
						double* lWeights = lCluster->GetControlPointWeights();
						for (int k = 0; k < lIndexCount; ++k)
						{
							subDeformer.indexes.push_back(lIndices[k]);
							subDeformer.weights.push_back(lWeights[k]);
						}

						FbxAMatrix lMatrix;
						lCluster->GetTransformMatrix(lMatrix);
						subDeformer.transform = Mat4x4(lMatrix.GetRow(0), lMatrix.GetRow(1), lMatrix.GetRow(2), lMatrix.GetRow(3));
						lCluster->GetTransformLinkMatrix(lMatrix);
						subDeformer.transformLink = Mat4x4(lMatrix.GetRow(0), lMatrix.GetRow(1), lMatrix.GetRow(2), lMatrix.GetRow(3));
						subDeformer.linkBoneUuid = lCluster->GetLink()->GetUniqueID();
						subDeformer.subDeformerName = std::string(lCluster->GetName());
						subDeformer.uuid = lCluster->GetUniqueID();

						mSubDeformers.push_back(subDeformer);
						mConnections.push_back(UInt64Vector2(lCluster->GetUniqueID(), lCluster->GetLink()->GetUniqueID()));
					}
				}
			}
		}
	}
}

void Importer::AnalyzeMesh(FbxNode* pNode)
{
	FbxMesh* lMesh = (FbxMesh*)pNode->GetNodeAttribute();

	mConnections.push_back(UInt64Vector2(pNode->GetUniqueID(), lMesh->GetUniqueID()));

	Mesh mesh(lMesh->GetName());
	mesh.uuid = lMesh->GetUniqueID();

	int lControlPointsCount = lMesh->GetControlPointsCount();
	FbxVector4* lControlPoints = lMesh->GetControlPoints();
	for (int i = 0; i < lControlPointsCount; ++i)
	{
		mesh.mVertices.push_back(Vector3(lControlPoints[i][0], lControlPoints[i][1], lControlPoints[i][2]));
	}

	FbxGeometryElementNormal* leNormal = lMesh->GetElementNormal();
	FbxGeometryElementUV* leUV = lMesh->GetElementUV();
	int lPolygonCount = lMesh->GetPolygonCount();
	int vertexId = 0;
	for (int i = 0; i < lPolygonCount; ++i)
	{
		int lPolygonSize = lMesh->GetPolygonSize(i);
		for (int j = 0; j < lPolygonSize; ++j)
		{
			int polyIndex = lMesh->GetPolygonVertex(i, j);
			if (j == lPolygonSize - 1)
			{
				polyIndex ^= -1;
			}
			mesh.mIndices.push_back(polyIndex);

			if (leUV != nullptr)
			{
				LayerElementUVInfo& uvInfo = GetUVInfo(0, leUV->GetName(), mesh);
				switch (leUV->GetMappingMode())
				{
				case FbxGeometryElement::eByControlPoint:
					switch (leUV->GetReferenceMode())
					{
					case FbxGeometryElement::eDirect:
						break;
					case FbxGeometryElement::eIndexToDirect:
						break;
					default:
						break;
					}
					break;
				case FbxGeometryElement::eByPolygonVertex:
					{
						int lTextureUVIndex = lMesh->GetTextureUVIndex(i, j);
						uvInfo.mUVIndices.push_back(lTextureUVIndex);
					}
					break;
				default:
					break;
				}
			}

			if(leNormal != nullptr)
			{
				if (leNormal->GetMappingMode() == FbxGeometryElement::eByPolygonVertex)
				{
					mesh.normalMapType = "ByPolygonVertex";
					FbxVector4 normal;
					int id;
					switch (leNormal->GetReferenceMode())
					{
					case FbxGeometryElement::eDirect:
						mesh.normalRefType = "Direct";
						normal = leNormal->GetDirectArray().GetAt(vertexId);
						break;
					case FbxGeometryElement::eIndexToDirect:
						mesh.normalRefType = "IndexToDirect";
						id = leNormal->GetIndexArray().GetAt(vertexId);
						normal = leNormal->GetDirectArray().GetAt(id);
						break;
					default:
						break; // other reference modes not shown here!
					}
					if (!normal.IsZero(3))
					{
						mesh.mNormals.push_back(Vector3(normal[0], normal[1], normal[2]));
					}
				}
			}
			vertexId++;
		}
	}

	if (leUV != nullptr)
	{
		LayerElementUVInfo& uvInfo = GetUVInfo(0, leUV->GetName(), mesh);
		int uvCount = leUV->GetDirectArray().GetCount();
		for (int i = 0; i < uvCount; ++i)
		{
			FbxVector2 uv = leUV->GetDirectArray().GetAt(i);
			uvInfo.mUVs.push_back(UV(uv[0], uv[1]));
		}
	}

	int edgeCount = lMesh->GetMeshEdgeCount();
	int startVerIndex = 0;
	int endVerIndex = 0;
	lMesh->BeginGetMeshEdgeVertices();
	for (int i = 0; i < edgeCount; ++i)
	{
		lMesh->GetMeshEdgeVertices(i, startVerIndex, endVerIndex);
		mesh.edges.push_back(IntVector2(startVerIndex, endVerIndex));
	}
	lMesh->EndGetMeshEdgeVertices();

	FbxGeometryElementSmoothing* leSmoothing = lMesh->GetElementSmoothing();
	if (leSmoothing != nullptr)
	{
		if (leSmoothing->GetReferenceMode() == FbxLayerElement::eDirect)
		{
			switch (leSmoothing->GetMappingMode())
			{
			case FbxLayerElement::eByPolygon:
				mesh.mSmoothMode = 0;
				break;
			case FbxLayerElement::eByEdge:
				mesh.mSmoothMode = 1;
				break;
			default:
				break;
			}
			int smoothingCount = leSmoothing->GetDirectArray().GetCount();
			for (int i = 0; i < smoothingCount; ++i)
			{
				mesh.mSmoothing.push_back(leSmoothing->GetDirectArray().GetAt(i));
			}
		}
	}

	FbxGeometryElementMaterial* leMat = lMesh->GetElementMaterial();
	if (leMat != nullptr)
	{
		mesh.matElemName = std::string(leMat->GetName());
		if (leMat->GetMappingMode() == FbxLayerElement::eByPolygon)
		{
			mesh.mMatMapping = 0;
		}
		else if (leMat->GetMappingMode() == FbxLayerElement::eAllSame)
		{
			mesh.mMatMapping = 1;
		}

		if (leMat->GetReferenceMode() == FbxGeometryElement::eIndexToDirect)
		{
			mesh.mMatRef = 2;
		}

		int lIndexArrayCount = leMat->GetIndexArray().GetCount();
		for (int i = 0; i < lIndexArrayCount; i++)
		{
			int matIndex = leMat->GetIndexArray().GetAt(i);
			mesh.mMatIndices.push_back(matIndex);
		}
	}
	
	mMesh.push_back(mesh);

	AnalyzeLink(lMesh);
}

void Importer::AnalyzeBone(FbxNode* pNode, Node& node)
{
	FbxSkeleton* lSkeleton = (FbxSkeleton*)pNode->GetNodeAttribute();
	if (lSkeleton->GetSkeletonType() == FbxSkeleton::eLimbNode || lSkeleton->GetSkeletonType() == FbxSkeleton::eLimb)
	{
		mConnections.push_back(UInt64Vector2(pNode->GetUniqueID(), lSkeleton->GetUniqueID()));
		Bone bone(lSkeleton->GetName());
		bone.uuid = lSkeleton->GetUniqueID();
		mBones.push_back(bone);

		node.isBone = true;
	}

	if (lSkeleton->GetSkeletonType() == FbxSkeleton::eLimbNode)
	{
		node.nodeAttributeName = std::string("LimbNode");
	}
	else if (lSkeleton->GetSkeletonType() == FbxSkeleton::eLimb)
	{
		node.nodeAttributeName = std::string("Limb");
	}
	else if (lSkeleton->GetSkeletonType() == FbxSkeleton::eRoot)
	{
		node.nodeAttributeName = std::string("Root");
	}
	
}

void Importer::AnalyzeContent(FbxNode* pNode)
{
	FbxNodeAttribute::EType lAttributeType;

	Node node(pNode->GetUniqueID(), pNode->LclTranslation.Get(), pNode->LclRotation.Get(), pNode->LclScaling.Get());
	node.nodeName = std::string(pNode->GetName());
	node.GeometricTranslation = Vector3(pNode->GeometricTranslation.Get()[0], pNode->GeometricTranslation.Get()[1], pNode->GeometricTranslation.Get()[2]);
	node.GeometricRotation = Vector3(pNode->GeometricRotation.Get()[0], pNode->GeometricRotation.Get()[1], pNode->GeometricRotation.Get()[2]);
	node.GeometricScaling = Vector3(pNode->GeometricScaling.Get()[0], pNode->GeometricScaling.Get()[1], pNode->GeometricScaling.Get()[2]);
	node.RotationOffset = Vector3(pNode->RotationOffset.Get()[0], pNode->RotationOffset.Get()[1], pNode->RotationOffset.Get()[2]);
	node.RotationPivot = Vector3(pNode->RotationPivot.Get()[0], pNode->RotationPivot.Get()[1], pNode->RotationPivot.Get()[2]);
	node.ScalingOffset = Vector3(pNode->ScalingOffset.Get()[0], pNode->ScalingOffset.Get()[1], pNode->ScalingOffset.Get()[2]);
	node.ScalingPivot = Vector3(pNode->ScalingPivot.Get()[0], pNode->ScalingPivot.Get()[1], pNode->ScalingPivot.Get()[2]);
	node.PreRotation = Vector3(pNode->PreRotation.Get()[0], pNode->PreRotation.Get()[1], pNode->PreRotation.Get()[2]);
	node.PostRotation = Vector3(pNode->PostRotation.Get()[0], pNode->PostRotation.Get()[1], pNode->PostRotation.Get()[2]);
	node.RotationOrder = pNode->RotationOrder.Get();
	node.RotationActive = pNode->RotationActive.Get();

	if (pNode->GetNodeAttribute() == NULL)
	{
		std::cout << "NULL Node Attribute" << std::endl;
	}
	else
	{
		lAttributeType = (pNode->GetNodeAttribute()->GetAttributeType());

		switch (lAttributeType)
		{
		case FbxNodeAttribute::eMesh:
			AnalyzeMesh(pNode);
			AnalyzeMaterial(pNode);
			node.nodeAttributeName = std::string("Mesh");
			break;
		case FbxNodeAttribute::eSkeleton:
			AnalyzeBone(pNode, node);
			break;
		case FbxNodeAttribute::eNull:
			node.nodeAttributeName = std::string("Null");
			break;
		default:
			break;
		}
	}

	mModels.push_back(node);

	for (int i = 0; i < pNode->GetChildCount(); i++)
	{
		mConnections.push_back(UInt64Vector2(pNode->GetUniqueID(), pNode->GetChild(i)->GetUniqueID()));
		AnalyzeContent(pNode->GetChild(i));
	}
}

void Importer::AnalyzePose(FbxScene* pScene)
{
	int lPoseCount = pScene->GetPoseCount();
	for (int i = 0; i < lPoseCount; ++i)
	{
		FbxPose* lPose = pScene->GetPose(i);
		if (lPose->IsBindPose())
		{
			for (int j = 0; j < lPose->GetCount(); ++j)
			{
				const FbxMatrix& mat = lPose->GetMatrix(j);
				PoseNode pose(lPose->GetNodeName(j).GetCurrentName(), Mat4x4(mat.GetRow(0), mat.GetRow(1), mat.GetRow(2), mat.GetRow(3)));
				FbxNode* node = lPose->GetNode(j);
				if (node != nullptr)
				{
					pose.refNodeUuid = node->GetUniqueID();
				}
				mPoses.push_back(pose);
			}
		}
		
	}
}

void Importer::AnalyzeAnimation(FbxScene* pScene)
{
	for (int i = 0; i < pScene->GetSrcObjectCount<FbxAnimStack>(); ++i)
	{
		FbxAnimStack* lAnimStack = pScene->GetSrcObject<FbxAnimStack>(i);
		int nbAnimLayers = lAnimStack->GetMemberCount<FbxAnimLayer>();
		mStacks.push_back(NameUUID(lAnimStack->GetName(), lAnimStack->GetUniqueID()));
		for (int j = 0; j < nbAnimLayers; ++j)
		{
			FbxAnimLayer* lAnimLayer = lAnimStack->GetMember<FbxAnimLayer>(j);
			mLayers.push_back(NameUUID(lAnimLayer->GetName(), lAnimLayer->GetUniqueID()));
			mConnections.push_back(UInt64Vector2(lAnimStack->GetUniqueID(), lAnimLayer->GetUniqueID()));
			AnalyzeAnimation(lAnimStack, lAnimLayer, pScene->GetRootNode());
		}
	}
}

void Importer::AnalyzeChannel(FbxAnimLayer* pAnimLayer, const char* channelName, FbxAnimCurve* pAnimCurve, FbxAnimCurveNode* pAnimCurveNode, Channel& channel)
{
	if (pAnimCurve && pAnimCurveNode)
	{
		channel.defaultValue = pAnimCurveNode->GetChannelValue(channelName, 0.0);
		int lKeyCount = pAnimCurve->KeyGetCount();
		for (int lCount = 0; lCount < lKeyCount; ++lCount)
		{
			FbxTime lKeyTime = pAnimCurve->KeyGetTime(lCount);
			double lKeyValue = pAnimCurve->KeyGetValue(lCount);
			channel.keys.push_back(Key(lKeyTime.Get(), lKeyValue));
		}
	}
}

void Importer::AnalyzeAnimation(FbxAnimStack* pAnimStack, FbxAnimLayer* pAnimLayer, FbxNode* pNode)
{
	ModelAnim anim;
	anim.modelName = pNode->GetName();
	anim.refModelUUID = pNode->GetUniqueID();
	anim.refLayerUUID = pAnimLayer->GetUniqueID();
	anim.refStackUUID = pAnimStack->GetUniqueID();

	AnalyzeChannel(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_X, pNode->LclTranslation.GetCurve(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_X), 
		pNode->LclTranslation.GetCurveNode(pAnimLayer, false), anim.channels[T_X]);
	AnalyzeChannel(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_Y, pNode->LclTranslation.GetCurve(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_Y),
		pNode->LclTranslation.GetCurveNode(pAnimLayer, false), anim.channels[T_Y]);
	AnalyzeChannel(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_Z, pNode->LclTranslation.GetCurve(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_Z),
		pNode->LclTranslation.GetCurveNode(pAnimLayer, false), anim.channels[T_Z]);

	AnalyzeChannel(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_X, pNode->LclRotation.GetCurve(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_X),
		pNode->LclRotation.GetCurveNode(pAnimLayer, false), anim.channels[R_X]);
	AnalyzeChannel(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_Y, pNode->LclRotation.GetCurve(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_Y),
		pNode->LclRotation.GetCurveNode(pAnimLayer, false), anim.channels[R_Y]);
	AnalyzeChannel(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_Z, pNode->LclRotation.GetCurve(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_Z),
		pNode->LclRotation.GetCurveNode(pAnimLayer, false), anim.channels[R_Z]);

	AnalyzeChannel(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_X, pNode->LclScaling.GetCurve(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_X),
		pNode->LclScaling.GetCurveNode(pAnimLayer, false), anim.channels[S_X]);
	AnalyzeChannel(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_Y, pNode->LclScaling.GetCurve(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_Y),
		pNode->LclScaling.GetCurveNode(pAnimLayer, false), anim.channels[S_Y]);
	AnalyzeChannel(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_Z, pNode->LclScaling.GetCurve(pAnimLayer, FBXSDK_CURVENODE_COMPONENT_Z),
		pNode->LclScaling.GetCurveNode(pAnimLayer, false), anim.channels[S_Z]);

	mAnims.push_back(anim);

	for (int lModelCount = 0; lModelCount < pNode->GetChildCount(); ++lModelCount)
	{
		AnalyzeAnimation(pAnimStack, pAnimLayer, pNode->GetChild(lModelCount));
	}
}

void Importer::AnalyzeContent(FbxScene* pScene)
{
	FbxNode* lNode = pScene->GetRootNode();

	if (lNode)
	{
		for (int i = 0; i < lNode->GetChildCount(); i++)
		{
			mConnections.push_back(UInt64Vector2(0, lNode->GetChild(i)->GetUniqueID()));
			AnalyzeContent(lNode->GetChild(i));
		}
	}
}

void Importer::AnalyzeGlobalSettings(FbxGlobalSettings* pGlobalSettings)
{
	const FbxSystemUnit& systemUnit = pGlobalSettings->GetSystemUnit();
	mGlobalSettings.UnitScaleFactor = systemUnit.GetScaleFactor();
	const FbxSystemUnit& oriSystemUnit = pGlobalSettings->GetOriginalSystemUnit();
	mGlobalSettings.OriginalUnitScaleFactor = oriSystemUnit.GetScaleFactor();
	mGlobalSettings.CustomFrameRate = pGlobalSettings->GetCustomFrameRate();
	mGlobalSettings.TimeMode = pGlobalSettings->GetTimeMode();

	std::string axisUp;
	std::string axisForward;
	const FbxAxisSystem& axisSystem = pGlobalSettings->GetAxisSystem();
	int upSign = -1;
	FbxAxisSystem::EUpVector up = axisSystem.GetUpVector(upSign);
	int frontSign = -1;
	FbxAxisSystem::EFrontVector front = axisSystem.GetFrontVector(frontSign);
	switch (up)
	{
	case FbxAxisSystem::eXAxis:
		axisUp = "X";
		if (front == FbxAxisSystem::eParityEven)
		{
			axisForward = "Y";
		}
		else
		{
			axisForward = "Z";
		}
		break;
	case FbxAxisSystem::eYAxis:
		axisUp = "Y";
		if (front == FbxAxisSystem::eParityEven)
		{
			axisForward = "X";
		}
		else
		{
			axisForward = "Z";
		}
		break;
	case FbxAxisSystem::eZAxis:
		axisUp = "Z";
		if (front == FbxAxisSystem::eParityEven)
		{
			axisForward = "X";
		}
		else
		{
			axisForward = "Y";
		}
		break;
	default:
		break;
	}

	if (upSign == -1)
	{
		axisUp.insert(0, "-");
	}

	if (frontSign == 1)
	{
		axisForward.insert(0, "-");
	}

	sprintf_s(mGlobalSettings.AxisUp, 3, "%s", axisUp.c_str());
	sprintf_s(mGlobalSettings.AxisForward, 3, "%s", axisForward.c_str());

}

void Importer::PrintNode()
{
	for (const Node& _node : mModels)
	{
		FBXUtil::PrintNode(_node);
	}
}

void Importer::PrintMesh()
{
	for (const Mesh& _mesh : mMesh)
	{
		FBXUtil::PrintMesh(_mesh);
	}

	for (const Material& mat : mMaterials)
	{
		PrintMaterial(mat);
	}

	std::cout << "Textures:" << std::endl;
	for (const Texture& tex : mTextures)
	{
		PrintTexture(tex);
	}
}

void Importer::PrintSkeleton()
{
	for (const Bone& _bone : mBones)
	{
		PrintBone(_bone);
	}

	for (const PoseNode& _pose : mPoses)
	{
		PrintPoseNode(_pose);
	}

	for (const SubDeformer& _subDeformer : mSubDeformers)
	{
		PrintSubDeformer(_subDeformer);
	}
}

void Importer::PrintAnimation()
{
	for (const ModelAnim& _anim : mAnims)
	{
		PrintModelAnim(_anim);
	}
}

extern "C"
{
	DLLEXPORT Importer* Importer_New() { return new Importer(); }
	DLLEXPORT bool Importer_Import(Importer* importer, char* name) { return importer->Import(name); }
	DLLEXPORT bool Importer_GetGlobalSettings(Importer* importer, FBXUtil::GlobalSettings* globalSettings) { return importer->GetGlobalSettings(globalSettings); }

	DLLEXPORT int Importer_GetConnectionCount(Importer* importer) { return importer->GetConnectionCount(); }
	DLLEXPORT bool Importer_GetConnections(Importer* importer, UInt64Vector2* pConnection, long connectionSize) { return importer->GetConnections(pConnection, connectionSize); }

	DLLEXPORT int Importer_GetModelCount(Importer* importer) { return importer->GetModelCount(); }
	DLLEXPORT bool Importer_GetModelTransformProp(Importer* importer, int index, ObjectTransformProp* prop) { return importer->GetModelTransformProp(index, prop); }
	DLLEXPORT FbxUInt64 Importer_GetModelUUID(Importer* importer, int index) { return importer->GetModelUUID(index); }
	DLLEXPORT const char* Importer_GetModelName(Importer* importer, int index) { return importer->GetModelName(index); }
	DLLEXPORT const char* Importer_GetModelAttributeName(Importer* importer, int index) { return importer->GetModelAttributeName(index); }
	DLLEXPORT bool Importer_IsModelBone(Importer* importer, int index) { return importer->IsModelBone(index); }

	DLLEXPORT FbxUInt64 Importer_GetMeshUUID(Importer* importer, int index) { return importer->GetMeshUUID(index); }
	DLLEXPORT const char* Importer_GetMeshName(Importer* importer, int index) { return importer->GetMeshName(index); }
	DLLEXPORT const char* Importer_GetUVInfoName(Importer* importer, int index, int uvIndex, LayerElementInfo* layerElemInfo) { return importer->GetUVInfoName(index, uvIndex, layerElemInfo); }
	DLLEXPORT int Importer_GetMeshCount(Importer* importer) { return importer->GetMeshCount(); }
	DLLEXPORT int Importer_GetMeshVerticeSize(Importer* importer, int index) { return importer->GetMeshVerticeSize(index); }
	DLLEXPORT int Importer_GetMeshUVIndiceSize(Importer* importer, int index, int uvIndex) { return importer->GetMeshUVIndiceSize(index, uvIndex); }
	DLLEXPORT int Importer_GetMeshUVVerticeSize(Importer* importer, int index, int uvIndex) { return importer->GetMeshUVVerticeSize(index, uvIndex); }
	DLLEXPORT bool Importer_GetMeshVertice(Importer* importer, int index, double* pVertice, long verticeSize) { return importer->GetMeshVertice(index, pVertice, verticeSize); }
	DLLEXPORT bool Importer_GetMeshNormals(Importer* importer, int index, double* pNormals, long normalSize, FBXUtil::LayerElementInfo *layerElemInfo) { return importer->GetMeshNormals(index, pNormals, normalSize, layerElemInfo); }
	DLLEXPORT bool Importer_GetMeshSmoothings(Importer* importer, int index, int* pSmoothings, long smoothingSize, FBXUtil::LayerElementInfo *layerElemInfo) { return importer->GetMeshSmoothings(index, pSmoothings, smoothingSize, layerElemInfo); }
	DLLEXPORT bool Importer_GetMeshMaterialInfo(Importer* importer, int index, int* pMatIndex, long indiceSize, LayerElementInfo* layerElemInfo) { return importer->GetMeshMaterialInfo(index, pMatIndex, indiceSize, layerElemInfo); }
	DLLEXPORT int Importer_GetMeshIndiceSize(Importer* importer, int index) { return importer->GetMeshIndiceSize(index); }
	DLLEXPORT int Importer_GetMeshNormalSize(Importer* importer, int index) { return importer->GetMeshNormalSize(index); }
	DLLEXPORT int Importer_GetMeshUVInfoSize(Importer* importer, int index) { return importer->GetMeshUVInfoSize(index); }
	DLLEXPORT int Importer_GetMeshSmoothingSize(Importer* importer, int index) { return importer->GetMeshSmoothingSize(index); }
	DLLEXPORT int Importer_GetMeshMatIndiceSize(Importer* importer, int index) { return importer->GetMeshMatIndiceSize(index); }
	DLLEXPORT int Importer_GetMeshEdgeSize(Importer* importer, int index) { return importer->GetMeshEdgeSize(index); }
	DLLEXPORT bool Importer_GetMeshIndice(Importer* importer, int index, int* pIndice, long indiceSize) { return importer->GetMeshIndice(index, pIndice, indiceSize); }
	DLLEXPORT bool Importer_GetMeshUVIndice(Importer* importer, int index, int uvIndex, int* pIndice, long indiceSize) { return importer->GetMeshUVIndice(index, uvIndex, pIndice, indiceSize); }
	DLLEXPORT bool Importer_GetMeshUVVertice(Importer* importer, int index, int uvIndex, double* pVertice, long verticeSize) { return importer->GetMeshUVVertice(index, uvIndex, pVertice, verticeSize); }
	DLLEXPORT bool Importer_GetMeshEdges(Importer* importer, int index, int* pEdges, long edgeSize) { return importer->GetMeshEdges(index, pEdges, edgeSize); }

	DLLEXPORT int Importer_GetMaterialCount(Importer* importer) { return importer->GetMaterialCount(); }
	DLLEXPORT FbxUInt64 Importer_GetMaterialUUID(Importer* importer, int index) { return importer->GetMaterialUUID(index); }
	DLLEXPORT const char* Importer_GetMaterialName(Importer* importer, int index) { return importer->GetMaterialName(index); }
	DLLEXPORT bool Importer_GetMaterialProps(Importer* importer, int index, Vector3* pEmissive, Vector3* pAmbient, Vector3* pDiffuse, MatProps* pExtra) 
	{ return importer->GetMaterialProps(index, pEmissive, pAmbient, pDiffuse, pExtra); }
	DLLEXPORT int Importer_GetTextureCount(Importer* importer) { return importer->GetTextureCount(); }
	DLLEXPORT FbxUInt64 Importer_GetTextureUUID(Importer* importer, int index) { return importer->GetTextureUUID(index); }
	DLLEXPORT const char* Importer_GetTextureName(Importer* importer, int index) { return importer->GetTextureName(index); }
	DLLEXPORT const char* Importer_GetTextureRelFileName(Importer* importer, int index) { return importer->GetTextureRelFileName(index); }
	DLLEXPORT const char* Importer_GetTextureFileName(Importer* importer, int index) { return importer->GetTextureFileName(index); }
	DLLEXPORT const char* Importer_GetTextureMatProp(Importer* importer, int index) { return importer->GetTextureMatProp(index); }
	DLLEXPORT bool Importer_GetTextureMapping(Importer* importer, int index, Vector3* pTranslation, Vector3* pRotation, Vector3* pScaling, IntVector2* pWrapMode) { return importer->GetTextureMapping(index, pTranslation, pRotation, pScaling, pWrapMode); }

	DLLEXPORT int Importer_GetBoneCount(Importer* importer) { return importer->GetBoneCount(); }
	DLLEXPORT FbxUInt64 Importer_GetBoneUUID(Importer* importer, int index) { return importer->GetBoneUUID(index); }
	DLLEXPORT const char* Importer_GetBoneName(Importer* importer, int index) { return importer->GetBoneName(index); }
	DLLEXPORT int Importer_GetPoseCount(Importer* importer) { return importer->GetPoseCount(); }
	DLLEXPORT FbxUInt64 Importer_GetRefBoneUUID(Importer* importer, int index) { return importer->GetRefBoneUUID(index); }
	DLLEXPORT bool Importer_GetPoseMatrix(Importer* importer, int index, double* pV, int matSize) { return importer->GetPoseMatrix(index, pV, matSize); }
	DLLEXPORT int Importer_GetClusterCount(Importer* importer) { return importer->GetClusterCount(); }
	DLLEXPORT FbxUInt64 Importer_GetClusterUUID(Importer* importer, int index) { return importer->GetClusterUUID(index); }
	DLLEXPORT const char* Importer_GetClusterName(Importer* importer, int index) { return importer->GetClusterName(index); }
	DLLEXPORT int Importer_GetClusterIndiceSize(Importer* importer, int index) { return importer->GetClusterIndiceSize(index); }
	DLLEXPORT bool Importer_GetClusterWeightIndice(Importer* importer, int index, int* pIndice, double* pWeight, long indiceSize) { return importer->GetClusterWeightIndice(index, pIndice, pWeight, indiceSize); }
	DLLEXPORT bool Importer_GetClusterTransforms(Importer* importer, int index, double* pTransform, double* pLinkTransform, int matSize) { return importer->GetClusterTransforms(index, pTransform, pLinkTransform, matSize); }
	DLLEXPORT int Importer_GetSkinCount(Importer* importer) { return importer->GetSkinCount(); }
	DLLEXPORT FbxUInt64 Importer_GetSkinUUID(Importer* importer, int index) { return importer->GetSkinUUID(index); }
	DLLEXPORT const char* Importer_GetSkinName(Importer* importer, int index) { return importer->GetSkinName(index); }

	DLLEXPORT int Importer_GetStackCount(Importer* importer) { return importer->GetStackCount(); }
	DLLEXPORT FbxUInt64 Importer_GetStackUUID(Importer* importer, int index) { return importer->GetStackUUID(index); }
	DLLEXPORT double Importer_GetAnimChannelDefaultValue(Importer* importer, FbxUInt64 stackUUID, FbxUInt64 layerUUID, FbxUInt64 boneUUID, ChannelType channel) { return importer->GetAnimChannelDefaultValue(stackUUID, layerUUID, boneUUID, channel); }
	DLLEXPORT const char* Importer_GetStackName(Importer* importer, int index) { return importer->GetStackName(index); }
	DLLEXPORT const char* Importer_GetLayerName(Importer* importer, FbxUInt64 uuid) { return importer->GetLayerName(uuid); }
	DLLEXPORT int Importer_GetKeyCount(Importer* importer, FbxUInt64 stackUUID, FbxUInt64 layerUUID, FbxUInt64 boneUUID, ChannelType channel) { return importer->GetKeyCount(stackUUID, layerUUID, boneUUID, channel); }
	DLLEXPORT bool Importer_GetKeyTimeValue(Importer* importer, FbxUInt64 stackUUID, FbxUInt64 layerUUID, FbxUInt64 boneUUID, ChannelType channel, FbxLongLong* pTimes, double* pValues, int keyCount)
	{ return importer->GetKeyTimeValue(stackUUID, layerUUID, boneUUID, channel, pTimes, pValues, keyCount); }
	DLLEXPORT bool Importer_AnimationExist(Importer* importer, FbxUInt64 stackUUID, FbxUInt64 layerUUID, FbxUInt64 boneUUID) { return importer->AnimationExist(stackUUID, layerUUID, boneUUID); }

	DLLEXPORT void Importer_PrintMesh(Importer* importer) { importer->PrintMesh(); }
	DLLEXPORT void Importer_PrintNode(Importer* importer) { importer->PrintNode(); }
	DLLEXPORT void Importer_PrintSkeleton(Importer* importer) { importer->PrintSkeleton(); }
	DLLEXPORT void Importer_PrintAnimation(Importer* importer) { importer->PrintAnimation(); }
}
