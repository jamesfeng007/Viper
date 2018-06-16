#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <cstdio>
#include <fbxsdk.h>
#include "FbxTools.h"

using namespace FBXUtil;

#define DLLEXPORT __declspec(dllexport)

class Importer
{
public:
	Importer();
	~Importer();

	bool Import(char* name);
	bool GetGlobalSettings(GlobalSettings* globalSettings);

	int GetConnectionCount() { return static_cast<int>(mConnections.size()); }
	bool GetConnections(UInt64Vector2* pConnection, long connectionSize);
	
	FbxUInt64 GetModelUUID(int index);
	int GetModelCount() { return static_cast<int>(mModels.size()); }
	bool GetModelTransformProp(int index, ObjectTransformProp* prop);
	const char* GetModelName(int index) { return mModels.at(index).nodeName.c_str(); }
	bool IsModelBone(int index) { return mModels.at(index).isBone; }

	int GetMeshCount() { return static_cast<int>(mMesh.size()); }
	FbxUInt64 GetMeshUUID(int index);
	int GetMeshVerticeSize(int index);
	int GetMeshIndiceSize(int index);
	int GetMeshNormalSize(int index);
	int GetMeshSmoothingSize(int index);
	const char* GetMeshName(int index) { return mMesh.at(index).mMeshName.c_str(); }
	bool GetMeshVertice(int index, double* pVertice, long verticeSize);
	bool GetMeshIndice(int index, int* pIndice, long indiceSize);
	bool GetMeshNormals(int index, double* pNormals, long normalSize, LayerElementInfo* layerElemInfo);
	bool GetMeshSmoothings(int index, int* pSmoothings, long smoothingSize, LayerElementInfo* layerElemInfo);
	int GetMeshEdgeSize(int index);
	bool GetMeshEdges(int index, int* pEdges, long edgeSize);
	int GetMeshUVInfoSize(int index);
	const char* GetUVInfoName(int index, int uvIndex, LayerElementInfo* layerElemInfo);
	int GetMeshUVIndiceSize(int index, int uvIndex);
	bool GetMeshUVIndice(int index, int uvIndex, int* pIndice, long indiceSize);
	int GetMeshUVVerticeSize(int index, int uvIndex);
	bool GetMeshUVVertice(int index, int uvIndex, double* pVertice, long verticeSize);
	int GetMeshMatIndiceSize(int index);
	bool GetMeshMaterialInfo(int index, int* pMatIndex, long indiceSize, LayerElementInfo* layerElemInfo);

	int GetMaterialCount() { return static_cast<int>(mMaterials.size()); }
	FbxUInt64 GetMaterialUUID(int index);
	const char* GetMaterialName(int index);
	bool GetMaterialProps(int index, Vector3* pEmissive, Vector3* pAmbient, Vector3* pDiffuse, MatProps* pExtra);

	int GetTextureCount() { return static_cast<int>(mTextures.size());	}
	FbxUInt64 GetTextureUUID(int index);
	const char* GetTextureName(int index);
	const char* GetTextureFileName(int index);
	const char* GetTextureRelFileName(int index);
	const char* GetTextureMatProp(int index);
	bool GetTextureMapping(int index, Vector3* pTranslation, Vector3* pRotation, Vector3* pScaling, IntVector2* pWrapMode);

	int GetBoneCount() { return static_cast<int>(mBones.size()); }
	FbxUInt64 GetBoneUUID(int index);
	const char* GetBoneName(int index);
	int GetPoseCount() { return static_cast<int>(mPoses.size()); }
	FbxUInt64 GetRefBoneUUID(int index);
	bool GetPoseMatrix(int index, double* pV, int matSize);
	int GetClusterCount() { return static_cast<int>(mSubDeformers.size()); }
	FbxUInt64 GetClusterUUID(int index);
	int GetClusterIndiceSize(int index);
	const char* GetClusterName(int index);
	bool GetClusterWeightIndice(int index, int* pIndice, double* pWeight, long indiceSize);
	bool GetClusterTransforms(int index, double* pTransform, double* pLinkTransform, int matSize);
	int GetSkinCount() { return static_cast<int>(mDeformers.size()); }
	FbxUInt64 GetSkinUUID(int index);
	const char* GetSkinName(int index);

	int GetStackCount() { return static_cast<int>(mStacks.size()); }
	FbxUInt64 GetStackUUID(int index);
	double GetAnimChannelDefaultValue(FbxUInt64 stackUUID, FbxUInt64 layerUUID, FbxUInt64 boneUUID, ChannelType channel);
	const char* GetStackName(int index);
	const char* GetLayerName(FbxUInt64 uuid);
	int GetKeyCount(FbxUInt64 stackUUID, FbxUInt64 layerUUID, FbxUInt64 boneUUID, ChannelType channel);
	bool GetKeyTimeValue(FbxUInt64 stackUUID, FbxUInt64 layerUUID, FbxUInt64 boneUUID, ChannelType channel, FbxLongLong* pTimes, double* pValues, int keyCount);

	void PrintMesh();
	void PrintNode();
	void PrintSkeleton();
	void PrintAnimation();

private:
	void AnalyzeGlobalSettings(FbxGlobalSettings* pGlobalSettings);
	void AnalyzeContent(FbxScene* pScene);
	void AnalyzeContent(FbxNode* pScene);
	void AnalyzeMesh(FbxNode* pNode);
	void AnalyzeLink(FbxGeometry* pGeometry);
	void AnalyzeMaterial(FbxNode* pNode);
	void AnalyzeTexture(FbxProperty& prop, FbxSurfaceMaterial* lMaterial);
	void AnalyzeBone(FbxNode* pNode);
	void AnalyzePose(FbxScene* pScene);
	void AnalyzeAnimation(FbxScene* pScene);
	void AnalyzeAnimation(FbxAnimStack* pAnimStack, FbxAnimLayer* pAnimLayer, FbxNode* pNode);
	void AnalyzeChannel(FbxAnimLayer* pAnimLayer, const char* channelName, FbxAnimCurve* pAnimCurve, FbxAnimCurveNode* pAnimCurveNode, Channel& channel);
	

private:
	Texture& GetTexture(FbxUInt64 uuid, bool exist);

	std::ofstream mLogFile;
	std::streambuf * mCoutbuf;
	GlobalSettings mGlobalSettings;

	std::vector<Node> mModels;
	std::vector<Mesh> mMesh;
	std::vector<Material> mMaterials;
	std::vector<Texture> mTextures;
	std::vector<UInt64Vector2> mConnections; //(parent, child)
	std::vector<Bone> mBones;
	std::vector<PoseNode> mPoses;
	std::vector<SubDeformer> mSubDeformers;
	std::vector<Deformer> mDeformers;
	std::vector<ModelAnim> mAnims;
	std::vector<NameUUID> mLayers;
	std::vector<NameUUID> mStacks;;
};