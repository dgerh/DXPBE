#include "Mesh.h"

Mesh::Mesh(std::string fileLocation, DXContext* context, ID3D12GraphicsCommandList6* cmdList, RenderPipeline* pipeline, XMFLOAT4X4 p_modelMatrix, bool doWireframeIndices) {
	loadMesh(fileLocation, doWireframeIndices);
    vertexBuffer = VertexBuffer(vertexPositions, (UINT)(vertices.size() * sizeof(XMFLOAT3)), (UINT)sizeof(XMFLOAT3));
    indexBuffer = IndexBuffer(indices, (UINT)(indices.size() * sizeof(unsigned int)));

    modelMatrix = p_modelMatrix;

    vbv = vertexBuffer.passVertexDataToGPU(*context, cmdList);
    ibv = indexBuffer.passIndexDataToGPU(*context, cmdList);

    //Transition both buffers to their usable states
    D3D12_RESOURCE_BARRIER barriers[2] = {};

    // Vertex buffer barrier
    barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[0].Transition.pResource = vertexBuffer.getVertexBuffer().Get();
    barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
    barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    // Index buffer barrier
    barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barriers[1].Transition.pResource = indexBuffer.getIndexBuffer().Get();
    barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_INDEX_BUFFER;
    barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

    cmdList->ResourceBarrier(2, barriers);
}

void Mesh::loadMesh(std::string fileLocation, bool doWireframeIndices) {
    std::ifstream file(fileLocation);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file " << fileLocation << std::endl;
        return;
    }

    std::string line;
    std::vector<XMFLOAT3> normals;
    std::vector<std::vector<int>> faces; // To store face indices

    // Skip the first few lines (header or unnecessary data)
    for (int i = 0; i < 4; ++i) {
        std::getline(file, line);
    }

    while (std::getline(file, line)) {
        if (line[0] == 'v' && line[1] == ' ') {
            //line is a vertex
            std::string temp = line.substr(2); //skip the "v " prefix
            std::stringstream ss(temp);
            std::vector<float> v;
            float value;

            while (ss >> value) {
                v.push_back(value);
            }
            Vertex newVert;
            newVert.pos = XMFLOAT3(v[0], v[1], v[2]);
            vertexPositions.push_back(XMFLOAT3(v[0], v[1], v[2]));
            vertices.push_back(newVert);
        }
        else if (line[0] == 'v' && line[1] == 'n') {
            //line is a normal
            std::string temp = line.substr(3); //skip the "vn " prefix
            std::stringstream ss(temp);
            std::vector<float> v;
            float value;

            while (ss >> value) {
                v.push_back(value);
            }
            normals.push_back(XMFLOAT3(v[0], v[1], v[2]));
        }
        else if (line[0] == 'f') {
            //line is a face
            std::string temp = line.substr(2); //skip the "f " prefix
            std::stringstream ss(temp);
            std::vector<int> vertexIndices;
            std::string s;
            int num1, num2, num3;
            char slash1, slash2;
            //man i love string steams :)
            while (ss >> num1 >> slash1 >> num2 >> slash2 >> num3) {
                //subtract 1 because obj isn't 0 indexed
                vertexIndices.push_back(num1 - 1);
            }

            faces.push_back(vertexIndices);
        }
    }

    for (int i = 0; i < vertices.size(); i++) {
        vertices[i].nor = normals[i];
    }

    if (!doWireframeIndices) {
        for (auto face : faces) {
            numTriangles++;
            indices.push_back(face[0]);
            indices.push_back(face[1]);
            indices.push_back(face[2]);
        }
    }
    else {
        //wireframe indices, only works for cubes
        std::vector<std::pair<int, int>> edges = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0},
            {4, 5}, {5, 6}, {6, 7}, {7, 4},
            {0, 4}, {1, 5}, {2, 6}, {3, 7}
        };

        for (const auto& edge : edges) {
            indices.push_back(edge.first);
            indices.push_back(edge.second);
            numTriangles++;
        }
    }

    file.close();
}

D3D12_INDEX_BUFFER_VIEW* Mesh::getIBV() {
    return &ibv;
}

D3D12_VERTEX_BUFFER_VIEW* Mesh::getVBV() {
    return &vbv;
}

XMFLOAT4X4* Mesh::getModelMatrix() {
    return &modelMatrix;
}

void Mesh::releaseResources() {
    vertexBuffer.releaseResources();
    indexBuffer.releaseResources();
}

UINT Mesh::getNumTriangles() {
    return numTriangles;
}
