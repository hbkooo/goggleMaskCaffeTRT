#ifndef GOGGLEMASKCLASSIFY_H
#define GOGGLEMASKCLASSIFY_H

#include <iostream>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include <caffe/caffe.hpp>
#include "tensorrt/trtgogglemask.h"

using namespace cv;
using namespace std;
using namespace caffe;

class GoggleMaskClassify
{
public:
    GoggleMaskClassify(string &model, string network="goggleMask");
    ~GoggleMaskClassify();

    //void detectBatchImages(vector<cv::Mat> imgs, float threshold, vector<float> &scales, vector<vector<FaceDetectInfo>> &faceInfos);
    void classify(Mat &img, std::vector<float> &results, float &scale);

private:
    void pre_process(Mat img, int inputW, int inputH, float &scale, Mat &resize);
    cv::Mat SetMean(const string& mean_value, int inputW, int inputH, int channels_);
private:
    boost::shared_ptr<caffe::Net<float>> Net_;

    TrtGoggleMaskNet *trtNet;
    float *cpuBuffers;

    string network;


#ifdef USE_NPP
    typedef struct GPUImg
    {
        void *data;
        int width;
        int height;
        int channel;
    } GPUImg;

    GPUImg _gpu_data8u;
    GPUImg _resize_gpu_data8u;
    GPUImg _resize_gpu_data32f;
#endif
};

#endif // RETINAFACE_H
