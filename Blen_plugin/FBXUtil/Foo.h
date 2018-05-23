#pragma once
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <fbxsdk.h>

#define DLLEXPORT __declspec(dllexport)

class Foo
{
public:
	Foo(int);
	void bar();
	int foobar(int);

	void AddVertex(float x, float y, float z);
	void AddNormal(float x, float y, float z);
	void CreateUVInfo(int uvIndex, char* name);
	void AddUV(int uvIndex, float x, float y);
	void AddIndex(int index);
	void AddUVIndex(int uvIndex, int index);
	void AddLoopStart(int start);
	void AddSmoothing(int smooth);
	void SetSmoothMode(int mode);
	void SetMeshName(char* name);
	void Print();

	bool Export(char* filePath);

	struct Vertex
	{
		Vertex() :x(0.f), y(0.f), z(0.f)
		{

		}

		Vertex(float _x, float _y, float _z)
			:x(_x), y(_y), z(_z)
		{

		}

		float x, y, z;
	};

	struct Normal
	{
		Normal() : x(0.f), y(0.f), z(0.f)
		{

		}

		Normal(float _x, float _y, float _z)
			:x(_x), y(_y), z(_z)
		{

		}
		float x, y, z;
	};

	struct UV
	{
		Normal() : x(0.f), y(0.f)
		{

		}

		Normal(float _x, float _y)
			:x(_x), y(_y)
		{

		}
		float x, y;
	};

	struct LayerElementUVInfo
	{
		LayerElementUVInfo()
		{

		}

		LayerElementUVInfo(int index, char* name)
			: uvIndex(index), name(std::string(name))
		{
			mUVs = std::vector<UV>();
			mUVIndices = std::vector<int>();
		}

		int uvIndex;
		std::string name;
		std::vector<UV> mUVs;
		std::vector<int> mUVIndices;
	};

private:

	bool CreateScene(FbxScene* pScene);
private:
	int val;
	std::string mMeshName;
	std::vector<Vertex> mVertices;
	std::vector<Normal> mNormals;
	std::vector<int> mIndices;
	std::vector<int> mLoopStart;
	std::vector<int> mSmoothing;
	std::map<int, LayerElementUVInfo> mUVInfos;
	 int mSmoothMode; //FbxLayerElement::EMappingMode: eByPolygon-0, eByEdge-1
};