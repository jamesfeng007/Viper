#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <fbxsdk.h>

namespace FBXUtil
{
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
	typedef Vector2<FbxUInt64> UInt64Vector2;

	struct Vector3
	{
		Vector3() :x(0.f), y(0.f), z(0.f)
		{

		}

		Vector3(double _x, double _y, double _z)
			:x(_x), y(_y), z(_z)
		{

		}

		double x, y, z;
	};

	struct Vector4
	{
		Vector4() :x(0.f), y(0.f), z(0.f), w(0.f)
		{

		}

		Vector4(double _x, double _y, double _z, double _w)
			:x(_x), y(_y), z(_z), w(_w)
		{

		}

		double x, y, z, w;
	};

	struct Node
	{
		Node()
		{

		}

		Node(FbxUInt64 _uuid, Vector3 _lclTranslation, Vector3 _lclRotation, Vector3 _lclScaling)
			: Node()
		{
			lclTranslation = _lclTranslation;
			lclRotation = _lclRotation;
			lclScaling = _lclScaling;
			uuid = _uuid;

		}

		Node(FbxUInt64 _uuid, FbxDouble3 _lclTranslation, FbxDouble3 _lclRotation, FbxDouble3 _lclScaling)
			: Node(_uuid, Vector3(_lclTranslation[0], _lclTranslation[1], _lclTranslation[2]),
				Vector3(_lclRotation[0], _lclRotation[1], _lclRotation[2]), Vector3(_lclScaling[0], _lclScaling[1], _lclScaling[2]))
		{

		}

		FbxUInt64 uuid;
		std::string nodeName;
		Vector3 lclTranslation;
		Vector3 lclRotation;
		Vector3 lclScaling;
		Vector3 GeometricTranslation;
		Vector3 GeometricRotation;
		Vector3 GeometricScaling;
		Vector3 RotationOffset;
		Vector3 RotationPivot;
		Vector3 ScalingOffset;
		Vector3 ScalingPivot;
		Vector3 PreRotation;
		Vector3 PostRotation;
		int RotationOrder;
		bool RotationActive;
	};

	struct Mat4x4
	{
		Mat4x4()
		{

		}

		Mat4x4(double _x0, double _x1, double _x2, double _x3,
			double _y0, double _y1, double _y2, double _y3,
			double _z0, double _z1, double _z2, double _z3,
			double _w0, double _w1, double _w2, double _w3) :
			x0(_x0), x1(_x1), x2(_x2), x3(_x3),
			y0(_y0), y1(_y1), y2(_y2), y3(_y3),
			z0(_z0), z1(_z1), z2(_z2), z3(_z3),
			w0(_w0), w1(_w1), w2(_w2), w3(_w3)
		{

		}

		double x0, x1, x2, x3;
		double y0, y1, y2, y3;
		double z0, z1, z2, z3;
		double w0, w1, w2, w3;
	};

	struct ObjectTransformProp
	{
		Vector3 lclTranslation;
		Vector3 lclRotation;
		Vector3 lclScaling;
		Vector3 GeometricTranslation;
		Vector3 GeometricRotation;
		Vector3 GeometricScaling;
		Vector3 RotationOffset;
		Vector3 RotationPivot;
		Vector3 ScalingOffset;
		Vector3 ScalingPivot;
		Vector3 PreRotation;
		Vector3 PostRotation;
		int RotationOrder;
		bool RotationActive;
	};

	struct LayerElementInfo
	{
		char MappingType[32];
		char RefType[32];
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

		UV(double _x, double _y)
			:x(_x), y(_y)
		{

		}
		double x, y;
	};

	struct LayerElementUVInfo
	{
		LayerElementUVInfo()
		{
			mUVs = std::vector<UV>();
			mUVIndices = std::vector<int>();
		}

		LayerElementUVInfo(int index, const char* n)
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

	struct Texture : Node
	{
		Texture()
		{
			parentMat = std::vector<std::string>();
		}

		Texture(char* _name, char* _fileName, char* _relFileName, int _alphaSource, bool _premultiplyAlpha, int _currentMappingType, char* _UVSet, int _wrapModeU, int _wrapModeV,
			Vector3 _translation, Vector3 _scaling, bool _useMaterial, bool _useMipMap)
			: name(std::string(_name)), fileName(std::string(_fileName)), relFileName(std::string(_relFileName)), alphaSource(_alphaSource), premultiplyAlpha(_premultiplyAlpha),
			currentMappingType(_currentMappingType), UVSet(std::string(_UVSet)), wrapModeU(_wrapModeU), wrapModeV(_wrapModeV), translation(_translation),
			scaling(_scaling), useMaterial(_useMaterial), useMipMap(_useMipMap)
		{
			parentMat = std::vector<std::string>();
		}

		std::string name;
		std::string fileName;
		std::string relFileName;
		int alphaSource;
		bool premultiplyAlpha;
		int currentMappingType;
		std::string UVSet;
		int wrapModeU;
		int wrapModeV;
		Vector3 translation;
		Vector3 scaling;
		bool useMaterial;
		bool useMipMap;
		std::vector<std::string> parentMat;
		std::string matProp;
	};

	struct Material : Node
	{
		Material()
		{

		}

		Material(char* mName, char* sName, Vector3 diffuse, Vector3 ambient, Vector3 emissive)
			: materialName(std::string(mName)), shadingName(std::string(sName))
			, emissiveColor(FbxDouble3(emissive.x, emissive.y, emissive.z))
			, ambientColor(FbxDouble3(ambient.x, ambient.y, ambient.z)), diffuseColor(FbxDouble3(diffuse.x, diffuse.y, diffuse.z))
		{

		}

		std::string materialName;
		std::string shadingName;
		FbxDouble3 emissiveColor;
		FbxDouble3 ambientColor;
		FbxDouble3 diffuseColor;
	};

	struct Mesh : Node
	{
		Mesh()
		{
			mVertices = std::vector<Vector3>();
			mIndices = std::vector<int>();
			mLoopStart = std::vector<int>();
			mNormals = std::vector<Vector3>();
			mSmoothing = std::vector<int>();
			mUVInfos = std::map<int, LayerElementUVInfo>();
			mMatIndices = std::vector<int>();
			mBinormals = std::vector<Vector3>();
			mTangents = std::vector<Vector3>();
			mSmoothMode = -1;
			mMatMapping = -1;
			mMatRef = -1;
		}

		Mesh(const char* name)
			: Mesh()
		{
			mMeshName = std::string(name);
		}

		std::string mMeshName;
		std::vector<Vector3> mVertices;
		std::vector<int> mIndices;
		std::vector<int> mLoopStart;
		std::vector<Vector3> mNormals;
		std::string normalMapType;
		std::string normalRefType;
		std::vector<int> mSmoothing;
		int mSmoothMode; //FbxLayerElement::EMappingMode: eByPolygon-0, eByEdge-1
		std::map<int, LayerElementUVInfo> mUVInfos;
		std::vector<IntVector2> edges;
		std::vector<int> mMatIndices;
		int mMatMapping; //FbxLayerElement::EMappingMode: eByPolygon-0, eAllSame-1
		int mMatRef; //FbxLayerElement::EReferenceMode: eDirect-0, eIndex-1, eIndexToDirect-2
		std::string matElemName;
		std::vector<Vector3> mBinormals;
		std::string binormalName;
		std::vector<Vector3> mTangents;
		std::string tangentName;
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
			weights = std::vector<double>();
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
		std::vector<double> weights;
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

		Key(double f, double v)
			: frame(f), value(v)
		{

		}

		double frame;
		double value;
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
		double localTimeSpan[2];
		double referenceTimeSpan[2];
		std::map<std::string, ModelAnim> models;
	};

	struct GlobalSettings
	{
		double UnitScaleFactor;
		double OriginalUnitScaleFactor;
		double CustomFrameRate;
		int TimeMode;
		char AxisUp[3];
		char AxisForward[3];
	};

	std::ostream& operator<<(std::ostream &os, const IntVector2& vec);
	std::ostream& operator<<(std::ostream &os, const Vector3& vec);
	std::ostream& operator<< (std::ostream &os, const Mat4x4& mat);

	Mesh& GetMesh(char* name, std::map<std::string, Mesh>& meshLoader);
	LayerElementUVInfo& GetUVInfo(int uvIndex, const char* name, Mesh& mesh);
	void PrintMesh(const Mesh& mesh);
	void PrintMaterial(const Material& mat);
	void PrintTexture(const Texture& tex);
}

