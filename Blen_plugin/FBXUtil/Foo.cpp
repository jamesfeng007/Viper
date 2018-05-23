#include "Foo.h"
#include "Common/Common.h"
#include <cmath>

Foo::Foo(int n)
{
	val = n;
	mVertices = std::vector<Foo::Vertex>();
	mNormals = std::vector<Foo::Normal>();
	mIndices = std::vector<int>();
	mLoopStart = std::vector<int>();
	mSmoothing = std::vector<int>();
	mUVInfos = std::map<int, LayerElementUVInfo>();
}

bool Foo::CreateScene(FbxScene* pScene)
{
	FbxNode* lRootNode = pScene->GetRootNode();
	FbxNode* lNode = FbxNode::Create(pScene, mMeshName.c_str());
	lRootNode->AddChild(lNode);

	FbxMesh* lMesh = FbxMesh::Create(pScene, mMeshName.c_str());

	lMesh->InitControlPoints(static_cast<int>(mVertices.size()));
	FbxVector4* lControlPoints = lMesh->GetControlPoints();
	for (int i=0; i<mVertices.size(); ++i)
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

	for (int i = 0; i < mLoopStart.size(); ++i)
	{
		int loopStart = mLoopStart[i];
		int loopEnd = i < mLoopStart.size() - 1 ? mLoopStart[i + 1] - 1 : static_cast<int>(mIndices.size()) - 1;
		lMesh->BeginPolygon();
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


	lNode->SetNodeAttribute(lMesh);

	return true;
}

void Foo::bar()
{
	std::cout << "Value is " << val << std::endl;
}

int Foo::foobar(int n)
{
	return val + n;
}

void Foo::CreateUVInfo(int uvIndex, char* name)
{
	Foo::LayerElementUVInfo uvInfo = LayerElementUVInfo(uvIndex, name);
	mUVInfos.insert(std::make_pair(uvIndex, uvInfo));
}

void Foo::AddVertex(float x, float y, float z)
{
	mVertices.push_back(Foo::Vertex(x, y, z));
}

void Foo::AddNormal(float x, float y, float z)
{
	mNormals.push_back(Foo::Normal(x, y, z));
}

void Foo::AddUV(int uvIndex, float x, float y)
{
	mUVInfos.at(uvIndex).mUVs.push_back(Foo::UV(x, y));
}

void Foo::AddUVIndex(int uvIndex, int index)
{
	mUVInfos.at(uvIndex).mUVIndices.push_back(index);
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

void Foo::SetMeshName(char* name)
{
	mMeshName = std::string(name);
}

void Foo::Print()
{
	for (Foo::Vertex v : mVertices)
	{
		std::cout << "vertex[ " << v.x << ", " << v.y << ", " << v.z << "]" << std::endl;
	}

	std::cout << "index[ ";
	for (int ix : mIndices)
	{
		std::cout << ix << ", ";
	}
	std::cout << " ]" << std::endl;

	for (Foo::Normal n : mNormals)
	{
		std::cout << "normal[ " << n.x << ", " << n.y << ", " << n.z << "]" << std::endl;
	}

	std::cout << "start[ ";
	for (int s : mLoopStart)
	{
		std::cout << s << ", ";
	}
	std::cout << " ]" << std::endl;

	std::cout << "mesh name: " << mMeshName << std::endl;

	std::cout << "smoothing mode:" << mSmoothMode << std::endl;
	for (int s : mSmoothing)
	{
		std::cout << s << ", ";
	}
	std::cout << std::endl;

	for (Foo::LayerElementUVInfo uvInfo : mUVInfos)
	{
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
}

bool Foo::Export(char* filePath)
{
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
	lResult = SaveScene(lSdkManager, lScene, filePath);

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
	DLLEXPORT Foo* Foo_new(int n) { return new Foo(n); }
	DLLEXPORT void Foo_bar(Foo* foo) { foo->bar(); }
	DLLEXPORT int Foo_foobar(Foo* foo, int n) { return foo->foobar(n); }
	DLLEXPORT void Foo_AddVertex(Foo* foo, float x, float y, float z) { foo->AddVertex(x, y, z); }
	DLLEXPORT void Foo_AddNormal(Foo* foo, float x, float y, float z) { foo->AddNormal(x, y, z); }
	DLLEXPORT void Foo_CreateUVInfo(Foo* foo, int uvIndex, char* name) { foo->CreateUVInfo(uvIndex, name); }
	DLLEXPORT void Foo_AddUV(Foo* foo, int uvIndex, float x, float y) { foo->AddUV(uvIndex, x, y); }
	DLLEXPORT void Foo_AddIndex(Foo* foo, int n) { foo->AddIndex(n); }
	DLLEXPORT void Foo_AddUVIndex(Foo* foo, int uvIndex, int n) { foo->AddUVIndex(uvIndex, n); }
	DLLEXPORT void Foo_AddLoopStart(Foo* foo, int n) { foo->AddLoopStart(n); }
	DLLEXPORT void Foo_AddSmoothing(Foo* foo, int s) { foo->AddSmoothing(s); }
	DLLEXPORT void Foo_SetSmoothMode(Foo* foo, int m) { foo->SetSmoothMode(m); }
	DLLEXPORT void Foo_SetMeshName(Foo* foo, char* name) { foo->SetMeshName(name); }
	DLLEXPORT bool Foo_Export(Foo* foo, char* name) { return foo->Export(name); }
	DLLEXPORT void Foo_Print(Foo* foo) { foo->Print(); }
}
