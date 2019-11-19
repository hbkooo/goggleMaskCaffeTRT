#include "trtgogglemask.h"
#include "trtutility.h"
#include <map>
#include <boost/bind/bind.hpp>
#include <assert.h>

using namespace std;

TrtGoggleMaskNet::TrtGoggleMaskNet(string netWorkName) : TrtNetBase(netWorkName)
{
    numBinding = 2;   // the number of input and output, classify task has one input(image) and one output(classify index)
    buffers = new void *[numBinding];
    for (int i = 0; i < numBinding; i++)
    {
        buffers[i] = NULL;
    }

    inputBuffer = NULL;

    workSpaceSize = 1 << 28;
    maxBatchSize = 8;
    useFp32 = true;

    outputs = {"prob"};
}

TrtGoggleMaskNet::~TrtGoggleMaskNet()
{
    destroyTrtContext();
    delete[] buffers;
}

void TrtGoggleMaskNet::doInference(float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context->getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 2);

    // create a stream

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    if(batchSize != 1) {
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));
        CHECK(cudaMalloc(&buffers[inputIndex], batchSize * inputSize));
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 4 * sizeof(float)));   // 4 is class number
    }

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * inputSize, cudaMemcpyHostToDevice, stream));
    context->enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 4* sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
}

void TrtGoggleMaskNet::allocateMemory(bool bUseCPUBuf)
{
    const ICudaEngine &cudaEngine = context->getEngine();
    assert(cudaEngine.getNbBindings() == numBinding);

    inputIndex = cudaEngine.getBindingIndex(inputBlobName.c_str());
    outputIndex = cudaEngine.getBindingIndex(outputs[0].c_str());

    inputDims = static_cast<DimsCHW &&>(cudaEngine.getBindingDimensions(inputIndex));

    // create GPU buffers and
    inputSize = 1 * inputDims.c() * inputDims.h() * inputDims.w() * sizeof(float);
    CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 4 * sizeof(float)));   // 4 is class number

}

void TrtGoggleMaskNet::releaseMemory(bool bUseCPUBuf)
{
    for (int i = 0; i < numBinding; i++)
    {
        if (buffers[i] != NULL)
        {
            CHECK(cudaFree(buffers[i]));
            buffers[i] = NULL;
        }
    }

}
