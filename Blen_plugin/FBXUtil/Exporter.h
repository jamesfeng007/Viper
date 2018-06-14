#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <fbxsdk.h>
#include "FbxTools.h"

#define DLLEXPORT __declspec(dllexport)

using namespace FBXUtil;

class Exporter
{
public:
	Exporter();

	void AddVertex(char* name, double x, double y, double z);
	void AddNormal(char* name, double x, double y, double z);
	void CreateUVInfo(char* meshName, int uvIndex, char* name);
	void AddUV(char* name, int uvIndex, double x, double y);
	void AddIndex(char* name, int index);
	void AddUVIndex(char* name, int uvIndex, int index);
	void AddMatIndex(char* name, int index);
	void AddTangent(char* name, Vector3 tangent);
	void AddBinormal(char* name, Vector3 binormal);
	void SetTangentName(char* name, char* tangentName);
	void SetBinormalName(char* name, char* binormalName);
	void AddLoopStart(char* name, int start);
	void AddSmoothing(char* name, int smooth);
	void SetSmoothMode(char* name, int mode);
	void SetMeshProperty(char* name, Vector3 trans, Vector3 rot, Vector3 sca);
	void AddMeshEdge(char* name, int startVertexIndex, int endVertexIndex);
	void AddMaterial(char* mName, char* sName, Vector3 diffuse, Vector3 ambient, Vector3 emissive);
	void AddBone(char* name, Vector3 lclTranslation, Vector3 lclRotation, Vector3 lclScaling);
	void AddBoneChild(char* child, char* parent);
	void AddPoseNode(char* name, Mat4x4 transform);
	void AddTexture(char* name, char* fileName, char* relFileName, int alphaSource, bool premultiplyAlpha, int currentMappingType, char* UVSet, int wrapModeU, int wrapModeV,
		Vector3 translation, Vector3 scaling, bool useMaterial, bool useMipMap);
	void SetTextureMatProp(char* name, char* matName, char* matProp);

	void AddSubDeformerIndex(char* mName, char* bName, int index);
	void AddSubDeformerWeight(char* mName, char* bName, double weight);
	void SetSubDeformerTransform(char* mName, char* bName, Mat4x4 transform, Vector4 quat);
	void SetSubDeformerTransformLink(char* mName, char* bName, Mat4x4 transformLink);

	void SetTimeSpan(char* takeName, double lStart, double lEnd, double rStart, double rEnd);
	void SetChannelDefaultValue(char* takeName, char* modelName, int type, double value);
	void AddChannelKey(char* takeName, char* modelName, int type, double frame, double value);
	void SetFPS(double fps);

	void SetAsASCII(bool asAscii) { mAsASCII = asAscii; }

	void PrintMeshProps(const Mesh& mesh);
	void PrintMesh();
	void PrintSkeleton();
	void PrintTakes();

	bool Export(char* filePath);



private:

	bool CreateScene(FbxScene* pScene);
	bool BuildMesh(FbxScene* pScene, FbxNode*& pMeshNode);
	bool BuildArmature(FbxScene* pScene, FbxNode*& pSkeletonNode);
	bool BuildDeformer(FbxScene* pScene, FbxNode* pMeshNode, FbxNode* pSkeletonNode);
	bool BuildPose(FbxScene* pScene, FbxNode*& pMeshNode, FbxNode* pSkeletonNode);
	bool BuildTakes(FbxScene* pScene, FbxNode* pSkeletonNode);

	
	SubDeformer& GetSubDeformer(const char* mName, const char* bName);
	ModelAnim& GetModelAnim(char* takeName, char* modelName);
	bool FillDeformer(FbxScene* pScene, FbxNode* pMeshNode, FbxNode* pSkeletonNode, FbxSkin* lSkin);
	bool FillDefaultValueKeys(FbxAnimCurveNode* animCurveNode, FbxAnimCurve* animCurve, const char* channelName, const Channel& channel);
private:
	std::ofstream mLogFile;
	std::streambuf * mCoutbuf;
	std::map<std::string, Mesh> mMesh;
	 std::vector<Material> mMaterials;
	 std::map<std::string, Bone> mBones;
	 std::map<std::string, Deformer> mDeformers;
	 std::map<std::string, PoseNode> mPoses;
	 std::map<std::string, Take> mTakes;
	 std::map<std::string, Texture> mTextures;
	 double mFps;
	 bool mAsASCII;
};