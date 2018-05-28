#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <fbxsdk.h>

#define DLLEXPORT __declspec(dllexport)

class Foo
{
public:
	Foo(int);

	template<typename T>
	struct Vector2
	{
		Vector2()
		{

		}

		Vector2(T _x, T _y)
			: x(_x), y(_y)
		{

		}

		T x, y;
	};

	typedef Vector2<int> IntVector2;

	struct Vector3
	{
		Vector3() :x(0.f), y(0.f), z(0.f)
		{

		}

		Vector3(float _x, float _y, float _z)
			:x(_x), y(_y), z(_z)
		{

		}

		float x, y, z;
	};

	struct Vector4
	{
		Vector4() :x(0.f), y(0.f), z(0.f), w(0.f)
		{

		}

		Vector4(float _x, float _y, float _z, float _w)
			:x(_x), y(_y), z(_z), w(_w)
		{

		}

		float x, y, z, w;
	};

	struct Mat4x4
	{
		Mat4x4()
		{

		}

		Mat4x4(float _x0, float _x1, float _x2, float _x3,
			float _y0, float _y1, float _y2, float _y3,
			float _z0, float _z1, float _z2, float _z3,
			float _w0, float _w1, float _w2, float _w3) :
			x0(_x0), x1(_x1), x2(_x2), x3(_x3), 
			y0(_y0), y1(_y1), y2(_y2), y3(_y3),
			z0(_z0), z1(_z1), z2(_z2), z3(_z3),
			w0(_w0), w1(_w1), w2(_w2), w3(_w3)
		{

		}

		float x0, x1, x2, x3;
		float y0, y1, y2, y3;
		float z0, z1, z2, z3;
		float w0, w1, w2, w3;
	};

	struct Pose
	{
		Pose()
		{

		}

		Pose(const char* name, Mat4x4 mat)
			: poseNodeName(std::string(name)), poseTransform(mat)
		{

		}

		std::string poseNodeName;
		Mat4x4 poseTransform;
	};

	struct UV
	{
		UV() : x(0.f), y(0.f)
		{

		}

		UV(float _x, float _y)
			:x(_x), y(_y)
		{

		}
		float x, y;
	};

	struct LayerElementUVInfo
	{
		LayerElementUVInfo()
		{
			mUVs = std::vector<UV>();
			mUVIndices = std::vector<int>();
		}

		LayerElementUVInfo(int index, char* n)
			: LayerElementUVInfo()
		{
			uvIndex = index;
			name = std::string(n);
		}

		int uvIndex;
		std::string name;
		std::vector<UV> mUVs;
		std::vector<int> mUVIndices;
	};

	struct Material
	{
		Material()
		{

		}

		Material(char* mName, char* sName)
			: materialName(std::string(mName)), shadingName(std::string(sName))
			, emissiveColor(FbxDouble3(0.0, 0.0, 0.0)), ambientColor(FbxDouble3(1.0, 0.0, 0.0)), diffuseColor(FbxDouble3(1.0, 1.0, 0.0))
		{

		}

		std::string materialName;
		std::string shadingName;
		FbxDouble3 emissiveColor;
		FbxDouble3 ambientColor;
		FbxDouble3 diffuseColor;
	};

	struct Mesh
	{
		Mesh()
		{

		}

		Mesh(char* mName, Vector3 _lclTranslation, Vector3 _lclRotation, Vector3 _lclScaling)
			: lclTranslation(_lclTranslation), lclRotation(_lclRotation), lclScaling(_lclScaling), mMeshName(std::string(mName))
		{

		}

		std::string mMeshName;
		Vector3 lclTranslation;
		Vector3 lclRotation;
		Vector3 lclScaling;
		std::vector<IntVector2> edges;
	};

	struct Bone
	{
		Bone()
		{
			parentName.clear();
		}

		Bone(char* name, Vector3 _lclTranslation, Vector3 _lclRotation, Vector3 _lclScaling)
			: Bone()
		{
			boneName = std::string(name);
			lclTranslation = _lclTranslation;
			lclRotation = _lclRotation;
			lclScaling = _lclScaling;
		}

		std::string boneName;
		Vector3 lclTranslation;
		Vector3 lclRotation;
		Vector3 lclScaling;
		std::string parentName;
	};

	struct SubDeformer
	{
		SubDeformer()
		{
			indexes = std::vector<int>();
			weights = std::vector<float>();
		}

		SubDeformer(const char* mName, const char* bName)
			: SubDeformer()
		{
			subDeformerName = std::string(mName) + std::string(" ") + std::string(bName);
			meshName = std::string(mName);
			boneName = std::string(bName);
		}

		std::string subDeformerName;
		std::string meshName;
		std::string boneName;
		std::vector<int> indexes;
		std::vector<float> weights;
		Mat4x4 transform;
		Mat4x4 transformLink;
		Vector4 quat;
	};

	struct Deformer
	{
		Deformer()
		{
			subDeformers = std::map<std::string, SubDeformer>();
		}

		Deformer(const char* mName)
			: Deformer()
		{
			meshName = std::string(mName);
			deformerName = meshName;
		}

		std::string deformerName;
		std::string meshName;
		std::map<std::string, SubDeformer> subDeformers;
	};

	struct Key
	{
		Key()
			: frame(0.f), value(0.f)
		{

		}

		Key(float f, float v)
			: frame(f), value(v)
		{

		}

		float frame;
		float value;
	};

	struct Channel
	{
		Channel()
			: defaultValue(0.0)
		{
			keys = std::vector<Key>();
		}
		double defaultValue;
		std::vector<Key> keys;
	};

	enum ChannelType
	{
		T_X = 0,
		T_Y,
		T_Z,
		R_X,
		R_Y,
		R_Z,
		S_X,
		S_Y,
		S_Z,
		ChannelMax
	};

	struct ModelAnim
	{
		ModelAnim()
		{

		}

		ModelAnim(char* mName)
			: modelName(std::string(mName))
		{

		}

		std::string modelName;
		Channel channels[ChannelMax];
	};

	struct Take
	{
		Take()
		{
			models = std::map<std::string, ModelAnim>();
		}

		Take(char* tName)
			: Take()
		{
			takeName = std::string(tName);
		}

		std::string takeName;
		float localTimeSpan[2];
		float referenceTimeSpan[2];
		std::map<std::string, ModelAnim> models;
	};

	void AddVertex(float x, float y, float z);
	void AddNormal(float x, float y, float z);
	void CreateUVInfo(int uvIndex, char* name);
	void AddUV(int uvIndex, float x, float y);
	void AddIndex(int index);
	void AddUVIndex(int uvIndex, int index);
	void AddMatIndex(int index);
	void AddLoopStart(int start);
	void AddSmoothing(int smooth);
	void SetSmoothMode(int mode);
	void SetMeshProperty(char* name, Vector3 trans, Vector3 rot, Vector3 sca);
	void AddMeshEdge(char* name, int startVertexIndex, int endVertexIndex);
	void AddMaterial(char* mName, char* sName);
	void AddBone(char* name, Vector3 lclTranslation, Vector3 lclRotation, Vector3 lclScaling);
	void AddBoneChild(char* child, char* parent);
	void AddPoseNode(char* name, Mat4x4 transform);

	void AddSubDeformerIndex(char* mName, char* bName, int index);
	void AddSubDeformerWeight(char* mName, char* bName, float weight);
	void SetSubDeformerTransform(char* mName, char* bName, Mat4x4 transform, Vector4 quat);
	void SetSubDeformerTransformLink(char* mName, char* bName, Mat4x4 transformLink);

	void SetTimeSpan(char* takeName, float lStart, float lEnd, float rStart, float rEnd);
	void SetChannelDefaultValue(char* takeName, char* modelName, int type, double value);
	void AddChannelKey(char* takeName, char* modelName, int type, float frame, float value);
	void SetFPS(float fps);

	void PrintMesh();
	void PrintSkeleton();
	void PrintTakes();

	bool Export(char* filePath);

	friend std::ostream& operator<<(std::ostream &os, const IntVector2& vec);
	friend std::ostream& operator<<(std::ostream &os, const Vector3& vec);
	friend std::ostream& operator<< (std::ostream &os, const Mat4x4& mat);

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
	int val;
	std::ofstream mLogFile;
	std::streambuf * mCoutbuf;
	Mesh mMesh;
	std::vector<Vector3> mVertices;
	std::vector<Vector3> mNormals;
	std::vector<int> mIndices;
	std::vector<int> mLoopStart;
	std::vector<int> mSmoothing;
	std::map<int, LayerElementUVInfo> mUVInfos;
	std::vector<int> mMatIndices;
	 int mSmoothMode; //FbxLayerElement::EMappingMode: eByPolygon-0, eByEdge-1
	 std::vector<Material> mMaterials;
	 std::map<std::string, Bone> mBones;
	 std::map<std::string, Deformer> mDeformers;
	 std::map<std::string, Pose> mPoses;
	 std::map<std::string, Take> mTakes;
	 float mFps;
};