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
	bool GetMaterialProps(int index, Vector3* pEmissive, Vector3* pAmbient, Vector3* pDiffuse);

	int GetTextureCount() { return static_cast<int>(mTextures.size());	}
	FbxUInt64 GetTextureUUID(int index);
	const char* GetTextureName(int index);
	const char* GetTextureFileName(int index);
	const char* GetTextureRelFileName(int index);
	const char* GetTextureMatProp(int index);
	bool GetTextureMapping(int index, Vector3* pTranslation, Vector3* pRotation, Vector3* pScaling, IntVector2* pWrapMode);

	void PrintMesh();
	void PrintNode();

private:
	void AnalyzeContent(FbxScene* pScene);
	void AnalyzeContent(FbxNode* pScene);
	void AnalyzeMesh(FbxNode* pNode);
	void AnalyzeMaterial(FbxNode* pNode);
	void AnalyzeTexture(FbxProperty& prop, FbxSurfaceMaterial* lMaterial);
	void AnalyzeGlobalSettings(FbxGlobalSettings* pGlobalSettings);

	

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
};