#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <cstdio>
#include <fbxsdk.h>
#include "FbxTools.h"

using namespace FBXUtil;

#define DLLEXPORT __declspec(dllexport)

class Importer
{
public:
	Importer();

	bool Import(char* name);
	bool GetGlobalSettings(GlobalSettings* globalSettings);
	bool GetMeshObjectTransformProp(int index, ObjectTransformProp* prop);
	int GetMeshCount() { return static_cast<int>(mMesh.size()); }
	int GetMeshVerticeSize(int index);
	int GetMeshIndiceSize(int index);
	int GetMeshNormalSize(int index);
	const char* GetMeshName(int index) { return mMesh.at(index).mMeshName.c_str(); }
	bool GetMeshVertice(int index, double* pVertice, long verticeSize);
	bool GetMeshIndice(int index, int* pIndice, long indiceSize);
	bool GetMeshNormals(int index, double* pNormals, long normalSize, LayerElementInfo *layerElemInfo);

	void PrintMesh();

private:
	void AnalyzeContent(FbxScene* pScene);
	void AnalyzeContent(FbxNode* pScene);
	void AnalyzeMesh(FbxNode* pNode);
	void AnalyzeGlobalSettings(FbxGlobalSettings* pGlobalSettings);

	

private:
	std::ofstream mLogFile;
	std::streambuf * mCoutbuf;
	GlobalSettings mGlobalSettings;

	std::vector<Mesh> mMesh;
};