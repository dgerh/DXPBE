#include "RenderPipeline.h"

RenderPipeline::RenderPipeline(std::string vertexShaderName, std::string fragShaderName, std::string rootSignatureShaderName, DXContext& context,
    CommandListID id, D3D12_DESCRIPTOR_HEAP_TYPE type, unsigned int numberOfDescriptors, D3D12_DESCRIPTOR_HEAP_FLAGS flags)
	: Pipeline(rootSignatureShaderName, context, id, type, numberOfDescriptors, flags), vertexShader(vertexShaderName), fragShader(fragShaderName) 
{
	createPSOD();
	createPipelineState(context.getDevice());
    getCommandList()->Close();
    context.resetCommandList(id);
}

D3D12_INPUT_ELEMENT_DESC vertexLayout[] =
{
    { "Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
};

void RenderPipeline::createPSOD() {
    gfxPsod.pRootSignature = rootSignature;
    gfxPsod.InputLayout.NumElements = _countof(vertexLayout);
    gfxPsod.InputLayout.pInputElementDescs = vertexLayout;
    gfxPsod.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;

    gfxPsod.VS.BytecodeLength = vertexShader.getSize();
    gfxPsod.VS.pShaderBytecode = vertexShader.getBuffer();

    gfxPsod.PS.BytecodeLength = fragShader.getSize();
    gfxPsod.PS.pShaderBytecode = fragShader.getBuffer();

    gfxPsod.DS.BytecodeLength = 0;
    gfxPsod.DS.pShaderBytecode = nullptr;
    gfxPsod.HS.BytecodeLength = 0;
    gfxPsod.HS.pShaderBytecode = nullptr;
    gfxPsod.GS.BytecodeLength = 0;
    gfxPsod.GS.pShaderBytecode = nullptr;
    gfxPsod.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    gfxPsod.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    gfxPsod.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    gfxPsod.RasterizerState.FrontCounterClockwise = FALSE;
    gfxPsod.RasterizerState.DepthBias = 0;
    gfxPsod.RasterizerState.DepthBiasClamp = .0f;
    gfxPsod.RasterizerState.SlopeScaledDepthBias = .0f;
    gfxPsod.RasterizerState.DepthClipEnable = FALSE;
    gfxPsod.RasterizerState.MultisampleEnable = FALSE;
    gfxPsod.RasterizerState.AntialiasedLineEnable = FALSE;
    gfxPsod.RasterizerState.ForcedSampleCount = 0;
    gfxPsod.StreamOutput.NumEntries = 0;
    gfxPsod.StreamOutput.NumStrides = 0;
    gfxPsod.StreamOutput.pBufferStrides = nullptr;
    gfxPsod.StreamOutput.pSODeclaration = nullptr;
    gfxPsod.StreamOutput.RasterizedStream = 0;

    gfxPsod.NumRenderTargets = 1;
    gfxPsod.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    gfxPsod.DSVFormat = DXGI_FORMAT_UNKNOWN;
    gfxPsod.BlendState.AlphaToCoverageEnable = FALSE;
    gfxPsod.BlendState.IndependentBlendEnable = FALSE;
    gfxPsod.BlendState.RenderTarget[0].LogicOpEnable = FALSE;
    //gfxPsod.BlendState.RenderTarget[0].BlendEnable = FALSE;
    //gfxPsod.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_ZERO;
    gfxPsod.BlendState.RenderTarget[0].BlendEnable = TRUE;
    gfxPsod.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_ONE;

    gfxPsod.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_ZERO;
    gfxPsod.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
    gfxPsod.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ZERO;
    gfxPsod.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
    gfxPsod.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
    gfxPsod.BlendState.RenderTarget[0].LogicOp = D3D12_LOGIC_OP_NOOP;
    gfxPsod.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    gfxPsod.DepthStencilState.DepthEnable = TRUE;
    gfxPsod.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    gfxPsod.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    gfxPsod.DepthStencilState.StencilEnable = FALSE;
    gfxPsod.DepthStencilState.StencilReadMask = 0;
    gfxPsod.DepthStencilState.StencilWriteMask = 0;
    gfxPsod.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    gfxPsod.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
    gfxPsod.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
    gfxPsod.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
    gfxPsod.DepthStencilState.BackFace.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    gfxPsod.DepthStencilState.BackFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
    gfxPsod.DepthStencilState.BackFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
    gfxPsod.DepthStencilState.BackFace.StencilPassOp = D3D12_STENCIL_OP_KEEP;
    gfxPsod.SampleMask = 0xFFFFFFFF;
    gfxPsod.SampleDesc.Count = 1;
    gfxPsod.SampleDesc.Quality = 0;

    gfxPsod.NodeMask = 0;
    gfxPsod.CachedPSO.CachedBlobSizeInBytes = 0;
    gfxPsod.CachedPSO.pCachedBlob = nullptr;
    gfxPsod.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
}

void RenderPipeline::createPipelineState(ComPointer<ID3D12Device6> device) {
	device->CreateGraphicsPipelineState(&gfxPsod, IID_PPV_ARGS(&pso));
}