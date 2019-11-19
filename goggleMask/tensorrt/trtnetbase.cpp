#include "trtnetbase.h"
#include "trtutility.h"
#include <assert.h>
#include <iterator>
#include <memory>
#include <sstream>
#include <zconf.h>

using namespace std;

//This function is used to trim space
string TrtNetBase::stringtrim(string s)
{
    int i = 0;
    while (s[i] == ' ')
    {
        i++;
    }
    s = s.substr(i);
    i = s.size() - 1;
    while (s[i] == ' ')
    {
        i--;
    }

    s = s.substr(0, i + 1);
    return s;
}

uint32_t TrtNetBase::getBatchSize() const
{
    return batchSize;
}

uint32_t TrtNetBase::getMaxBatchSize() const
{
    return maxBatchSize;
}

int TrtNetBase::getNetWidth() const
{
    return netWidth;
}

int TrtNetBase::getNetHeight() const
{
    return netHeight;
}

int TrtNetBase::getChannel() const
{
    return channel;
}

void *&TrtNetBase::getBuffer(const int &index)
{
    assert(index >= 0 && index < numBinding);
    return buffers[index];
}

float *&TrtNetBase::getInputBuf()
{
    return inputBuffer;
}

void TrtNetBase::setForcedFp32(const bool &forcedFp32)
{
    useFp32 = forcedFp32;
}

void TrtNetBase::setDumpResult(const bool &dumpResult)
{
    this->dumpResult = dumpResult;
}

void TrtNetBase::setTrtProfilerEnabled(const bool &enableTrtProfiler)
{
    this->enableTrtProfiler = enableTrtProfiler;
}

TrtNetBase::TrtNetBase(string netWorkName)
{
    pLogger = new Logger();
    profiler = new Profiler();
    runtime = NULL;
    engine = NULL;
    context = NULL;

    batchSize = 0;
    channel = 0;
    netWidth = 0;
    netHeight = 0;

    useFp32 = false;

    dumpResult = false;
    resultFile = "result.txt";
    enableTrtProfiler = false;
    this->netWorkName = netWorkName;
}

TrtNetBase::~TrtNetBase()
{
    delete pLogger;
    delete profiler;
}

bool TrtNetBase::parseNet(const string &deployfile)
{
    ifstream readfile;
    string line;
    readfile.open(deployfile, ios::in);
    if (!readfile)
    {
        printf("the deployfile doesn't exist!\n");
        return false;
    }

    while (1)
    {
        getline(readfile, line);
        string::size_type index;

        index = line.find("input_param");
        if (index == std::string::npos)
        {
            continue;
        }

        index = line.find("dim", 0);
        while (index == std::string::npos) {
            getline(readfile, line);
            index = line.find("dim", 0);
        }

//        getline(readfile, line);
//        index = line.find("dim:", 0);

        string dims = line.substr(index);

//        std::cout << line << std::endl;  //  input_param { shape: { dim: 1 dim: 3 dim: 32 dim: 32 } }
//        std::cout << dims << std::endl;  // dim: 1 dim: 3 dim: 32 dim: 32 } }

        vector<string> res;
        string result;
        stringstream input(dims);
        while(input >> result)
            res.push_back(result);

        string first = res[1];
        string second = res[3];
        string third = res[5];
        string fourth = res[7];

//        string first = line.substr(index + 5);
//        string second = line.substr(index + 12);
//        string third = line.substr(index + 19);
//        string fourth = line.substr(index + 28);


        batchSize = atoi(stringtrim(first).c_str());
        assert(batchSize > 0);

        channel = atoi(stringtrim(second).c_str());
        assert(channel > 0);

        netHeight = atoi(stringtrim(third).c_str());
        assert(netHeight > 0);

        netWidth = atoi(stringtrim(fourth).c_str());
        assert(netWidth > 0);

        break;
    }

    printf("batchSize:%d, channel:%d, netHeight:%d, netWidth:%d.\n", batchSize, channel, netHeight, netWidth);

    readfile.close();

    return true;
}


void TrtNetBase::buildTrtContext(const std::string &deployfile, const std::string &modelfile, const std::string &engineFile,
        bool bUseCPUBuf)
{
    if (!parseNet(deployfile))
    {
        printf("parse net failed, exit!\n");
        exit(0);
    }
//    std::cout << "loading build context test1 ...\n";

    if (access(engineFile.c_str(), 0) != 0) {
        caffeToTRTModel(deployfile, modelfile, engineFile, NULL);
        runtime = createInferRuntime(*pLogger);
        engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), NULL);
        trtModelStream->destroy();
    } else {
        engine = loadGIEEngine(engineFile);
    }

//    std::cout << "\n\n\n\nloading build context test5 ...\n";
    context = engine->createExecutionContext();
//    std::cout << "\n\n\n\nloading build context test6 ...\n";
    context->setProfiler(profiler);
//    std::cout << "\n\n\n\nloading build context test7 ...\n";
    allocateMemory(bUseCPUBuf);
//    std::cout << "\n\n\n\nloading build context test8 ...\n";
}

void TrtNetBase::destroyTrtContext(bool bUseCPUBuf)
{
    releaseMemory(bUseCPUBuf);
    context->destroy();
    engine->destroy();
    runtime->destroy();
}

void TrtNetBase::caffeToTRTModel(const std::string &deployFile, const std::string &modelFile, const std::string &engineFile,
                                 nvcaffeparser1::IPluginFactory *pluginFactory)
{
    // create API root class - must span the lifetime of the engine usage
//    std::cout << "loading build context test1.1 ...\n";
    IBuilder *builder = createInferBuilder(*pLogger);
    INetworkDefinition *network = builder->createNetwork();
//    std::cout << "loading build context test1.2 ...\n";

    // parse the caffe model to populate the network, then set the outputs
    ICaffeParser *parser = createCaffeParser();
    parser->setPluginFactory(pluginFactory);
//    std::cout << "loading build context test1.3 ...\n";

    bool useFp16 = builder->platformHasFastFp16();
    // if user specify
    if (useFp32)
    {
        useFp16 = 0;
    }
    std::cout << "loading build context test1.4 ...\n";
    DataType modelDataType = useFp16 ? DataType::kHALF : DataType::kFLOAT; // create a 16-bit model if it's natively supported
    // network definition that the parser will populate
    std::cout << "loading build context test1.4.1 ...\n";
//    std::cout << "loading build context deployFile : " << deployFile << std::endl;
//    std::cout << "loading build context modelFile : " << modelFile << std::endl;
    const IBlobNameToTensor *blobNameToTensor = parser->parse(deployFile.c_str(), modelFile.c_str(), *network, modelDataType);

//    std::cout << "loading build context test1.4.2 ...\n";
    assert(blobNameToTensor != nullptr);
    // the caffe file has no notion of outputs, so we need to manually say which tensors the engine should generate

    std::cout << "loading build context test1.5 ...\n";
    for (auto &s : outputs)
    {
        network->markOutput(*blobNameToTensor->find(s.c_str()));
        printf("outputs %s\n", s.c_str());
    }

//    std::cout << "loading build context test1.6 ...\n";
    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(workSpaceSize);

    // Eliminate the side-effect from the delay of GPU frequency boost
    //builder->setMinFindIterations(3);
    //builder->setAverageFindIterations(2);

//    std::cout << "loading build context test1.7 ...\n";
    ICudaEngine *engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();
//    std::cout << "loading build context test1.8 ...\n";

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();

    std::stringstream gieModelStream;
    gieModelStream.seekg(0,gieModelStream.beg);
    gieModelStream.write(static_cast<const char*>(trtModelStream->data()), trtModelStream->size());
    std::ofstream outFile;
    outFile.open(engineFile);
    outFile << gieModelStream.rdbuf();
    outFile.close();


    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}

nvinfer1::ICudaEngine* TrtNetBase::loadGIEEngine(const std::string planFilePath)
{
    // reading the model in memory
    std::cout << "Loading TRT Engine..." << std::endl;
//    assert(fileExists(planFilePath));
    std::stringstream gieModelStream;
    gieModelStream.seekg(0, gieModelStream.beg);
    std::ifstream cache(planFilePath);
    assert(cache.good());
    gieModelStream << cache.rdbuf();
    cache.close();

    // calculating model size
    gieModelStream.seekg(0, std::ios::end);
    const int modelSize = gieModelStream.tellg();
    gieModelStream.seekg(0, std::ios::beg);
    void* modelMem = malloc(modelSize);
    gieModelStream.read((char*) modelMem, modelSize);

    runtime = nvinfer1::createInferRuntime(*pLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelMem, modelSize, nullptr);
    free(modelMem);
    std::cout << "Loading Complete!" << std::endl;
    return engine;
}
