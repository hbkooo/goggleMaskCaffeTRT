#ifndef TRTGOGGLEMASKNET_H
#define TRTGOGGLEMASKNET_H

#include <fstream>
#include <iostream>
#include <cuda_runtime_api.h>
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "trtnetbase.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

class TrtGoggleMaskNet : public TrtNetBase
{
public:
    TrtGoggleMaskNet(std::string netWorkName);
    ~TrtGoggleMaskNet();

    /**
     *	@brief  doInference	            TensorRT推理函数
     *   @param  input		            数据输入
     *   @param  output		            result
     *   @param  batchSize		        批量数
     *   @return
     *
     *   @note
     */
    void doInference(float* input, float* output, int batchSize=1);

    void doInference(int batchSize, float *input = NULL) override {};

private:
    /**
    *	@brief  allocateMemory	        开辟内存空间
    *   @param  bUseCPUBuf		        是否使用CPU buffer
    *   @return 
    *
    *   @note					        
    */
    virtual void allocateMemory(bool bUseCPUBuf) override;

    /**
    *	@brief  releaseMemory	        释放内存空间
    *   @param  bUseCPUBuf		        是否使用CPU buffer
    *   @return 
    *
    *   @note					        
    */
    virtual void releaseMemory(bool bUseCPUBuf) override;

private:
    std::string inputBlobName = "data";

    int inputIndex, outputIndex;
    DimsCHW inputDims;
    size_t inputSize;

};

#endif // TRTGOGGLEMASKNET_H
