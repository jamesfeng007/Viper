#include "Importer.h"
#include "Common/Common.h"

Importer::Importer()
{
	mLogFile = std::ofstream("ImportFBXSdk.log");
	mCoutbuf = std::cout.rdbuf(); //save old buf
	std::cout.rdbuf(mLogFile.rdbuf()); //redirect std::cout to out.txt!

	mMesh = std::vector<Mesh>();
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
	}


	// Destroy all objects created by the FBX SDK.
	DestroySdkObjects(lSdkManager, lResult);

	std::cout.rdbuf(mCoutbuf); //reset to standard output again
	mLogFile.flush();
	mLogFile.close();

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

bool Importer::GetMeshObjectTransformProp(int index, ObjectTransformProp* prop)
{
	Mesh& mesh = mMesh.at(index);
	if (prop == nullptr)
	{
		return false;
	}

	prop->lclTranslation = mesh.lclTranslation;
	prop->lclRotation = mesh.lclRotation;
	prop->lclScaling = mesh.lclScaling;
	prop->GeometricTranslation = mesh.GeometricTranslation;
	prop->GeometricRotation = mesh.GeometricRotation;
	prop->GeometricScaling = mesh.GeometricScaling;
	prop->RotationOffset = mesh.RotationOffset;
	prop->RotationPivot = mesh.RotationPivot;
	prop->ScalingOffset = mesh.ScalingOffset;
	prop->ScalingPivot = mesh.ScalingPivot;
	prop->PreRotation = mesh.PreRotation;
	prop->PostRotation = mesh.PostRotation;
	prop->RotationOrder = mesh.RotationOrder;
	prop->RotationActive = mesh.RotationActive;

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

void Importer::AnalyzeMesh(FbxNode* pNode)
{
	std::cout << "Mesh Name: " << pNode->GetName() << std::endl;
	
	
	Mesh mesh(pNode->GetName(), pNode->LclTranslation.Get(), pNode->LclRotation.Get(), pNode->LclScaling.Get());
	mesh.GeometricTranslation = Vector3(pNode->GeometricTranslation.Get()[0], pNode->GeometricTranslation.Get()[1], pNode->GeometricTranslation.Get()[2]);
	mesh.GeometricRotation = Vector3(pNode->GeometricRotation.Get()[0], pNode->GeometricRotation.Get()[1], pNode->GeometricRotation.Get()[2]);
	mesh.GeometricScaling = Vector3(pNode->GeometricScaling.Get()[0], pNode->GeometricScaling.Get()[1], pNode->GeometricScaling.Get()[2]);
	mesh.RotationOffset = Vector3(pNode->RotationOffset.Get()[0], pNode->RotationOffset.Get()[1], pNode->RotationOffset.Get()[2]);
	mesh.RotationPivot = Vector3(pNode->RotationPivot.Get()[0], pNode->RotationPivot.Get()[1], pNode->RotationPivot.Get()[2]);
	mesh.ScalingOffset = Vector3(pNode->ScalingOffset.Get()[0], pNode->ScalingOffset.Get()[1], pNode->ScalingOffset.Get()[2]);
	mesh.ScalingPivot = Vector3(pNode->ScalingPivot.Get()[0], pNode->ScalingPivot.Get()[1], pNode->ScalingPivot.Get()[2]);
	mesh.PreRotation = Vector3(pNode->PreRotation.Get()[0], pNode->PreRotation.Get()[1], pNode->PreRotation.Get()[2]);
	mesh.PostRotation = Vector3(pNode->PostRotation.Get()[0], pNode->PostRotation.Get()[1], pNode->PostRotation.Get()[2]);
	mesh.RotationOrder = pNode->RotationOrder.Get();
	mesh.RotationActive = pNode->RotationActive.Get();


	FbxMesh* lMesh = (FbxMesh*)pNode->GetNodeAttribute();

	int lControlPointsCount = lMesh->GetControlPointsCount();
	FbxVector4* lControlPoints = lMesh->GetControlPoints();
	for (int i = 0; i < lControlPointsCount; ++i)
	{
		mesh.mVertices.push_back(Vector3(lControlPoints[i][0], lControlPoints[i][1], lControlPoints[i][2]));
	}

	FbxGeometryElementNormal* leNormal = lMesh->GetElementNormal();
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
	
	mMesh.push_back(mesh);
}

void Importer::AnalyzeContent(FbxNode* pNode)
{
	FbxNodeAttribute::EType lAttributeType;

	std::cout << "node uuid: " << pNode->GetUniqueID() << std::endl;

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
			break;
		default:
			break;
		}
	}

	for (int i = 0; i < pNode->GetChildCount(); i++)
	{
		AnalyzeContent(pNode->GetChild(i));
	}
}

void Importer::AnalyzeContent(FbxScene* pScene)
{
	FbxNode* lNode = pScene->GetRootNode();

	if (lNode)
	{
		for (int i = 0; i < lNode->GetChildCount(); i++)
		{
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

void Importer::PrintMesh()
{
	for (const Mesh& _mesh : mMesh)
	{
		FBXUtil::PrintMesh(_mesh);
	}
}

extern "C"
{
	DLLEXPORT Importer* Importer_New() { return new Importer(); }
	DLLEXPORT bool Importer_Import(Importer* importer, char* name) { return importer->Import(name); }
	DLLEXPORT bool Importer_GetGlobalSettings(Importer* importer, FBXUtil::GlobalSettings* globalSettings) { return importer->GetGlobalSettings(globalSettings); }
	DLLEXPORT const char* Importer_GetMeshName(Importer* importer, int index) { return importer->GetMeshName(index); }
	DLLEXPORT int Importer_GetMeshCount(Importer* importer) { return importer->GetMeshCount(); }
	DLLEXPORT int Importer_GetMeshVerticeSize(Importer* importer, int index) { return importer->GetMeshVerticeSize(index); }
	DLLEXPORT bool Importer_GetMeshVertice(Importer* importer, int index, double* pVertice, long verticeSize) { return importer->GetMeshVertice(index, pVertice, verticeSize); }
	DLLEXPORT bool Importer_GetMeshNormals(Importer* importer, int index, double* pNormals, long normalSize, FBXUtil::LayerElementInfo *layerElemInfo) { return importer->GetMeshNormals(index, pNormals, normalSize, layerElemInfo); }
	DLLEXPORT int Importer_GetMeshIndiceSize(Importer* importer, int index) { return importer->GetMeshIndiceSize(index); }
	DLLEXPORT int Importer_GetMeshNormalSize(Importer* importer, int index) { return importer->GetMeshNormalSize(index); }
	DLLEXPORT bool Importer_GetMeshIndice(Importer* importer, int index, int* pIndice, long indiceSize) { return importer->GetMeshIndice(index, pIndice, indiceSize); }
	DLLEXPORT bool Importer_GetMeshObjectTransformProp(Importer* importer, int index, ObjectTransformProp* prop) { return importer->GetMeshObjectTransformProp(index, prop); }

	DLLEXPORT void Importer_PrintMesh(Importer* importer) { importer->PrintMesh(); }
}
