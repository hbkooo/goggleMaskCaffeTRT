#include <iostream>
#include <RetinaFace.h>
#include <goggleMaskClassify.h>
#include "timer.h"

using namespace std;

#include <vector>
#include <string>
#include <iostream>
using namespace std;

vector<string> split(const string &s, const string &seperator){
    vector<string> result;
    typedef string::size_type string_size;
    string_size i = 0;

    while(i != s.size()){
        //找到字符串中首个不等于分隔符的字母；
        int flag = 0;
        while(i != s.size() && flag == 0){
            flag = 1;
            for(string_size x = 0; x < seperator.size(); ++x)
            if(s[i] == seperator[x]){
                ++i;
                flag = 0;
                 break;
                }
        }

        //找到又一个分隔符，将两个分隔符之间的字符串取出；
        flag = 0;
        string_size j = i;
        while(j != s.size() && flag == 0){
            for(string_size x = 0; x < seperator.size(); ++x)
                if(s[j] == seperator[x]){
                    flag = 1;
                 break;
                }
            if(flag == 0)
            ++j;
        }
        if(i != j){
            result.push_back(s.substr(i, j-i));
            i = j;
        }
    }
    return result;
}

int endsWith(string s,string sub){
    return s.rfind(sub)==(s.length()-sub.length())?1:0;
}

int classify(GoggleMaskClassify *classifier, const char* img_path){

    stringstream inputs(img_path);
    string res;
    std::vector<string> result;
    while(inputs >> res)
        result.push_back(res);



    cv::Mat img = cv::imread(result[0]);

    std::cout << "\ndetect image name : " << img_path << std::endl;
//    std::cout << "\nimg.shape( " << img.cols << ", " << img.rows << " )" << std::endl;

    float scale =1.0;

    std::vector<float> output;
    classifier->classify(img, output, scale);

    string text = "goggle_mask";
    float max = output[0];
    int max_index = 0;
    for (int i = 1; i < 4; i++) {
        if(output[i] > max) {
            max = output[i];
            max_index = i;
            if(i == 1) text = "goggle";
            else if(i == 2) text = "mask";
            else if (i == 3) text = "no";
        }
    }

    int count = 0;

    if(result.size() > 1) {

        int label = atoi(result[1].c_str());
        std::cout << "true label : " << label << ", detect label : " << max_index;
        if (label == max_index) {
            count = 1;
            std::cout << ", detect true , ";
        } else {
            std::cout << ", detect false , ";
        }


    }

    std::cout << "detect result : " << text << endl;

//    std::cout << "classify result : goggle_mask : " << output[0] << "\tgoggle : " << output[1]
//              << "\tmask : " << output[2] << "\tno : " << output[3] << std::endl;
    return count;
}


void detectFace(RetinaFace *faceDetector, GoggleMaskClassify *classifier, const char* img_path)
{
    std::cout << "\ndetect image name : " << img_path << std::endl;
    cv::Mat img = cv::imread(img_path);

	std::cout << "img.shape( " << img.cols << ", " << img.rows << " )" << std::endl;

    float scale =1.0, scale_classify = 1.0;
    std::vector<FaceDetectInfo> faceInfo;
    std::vector<float> classify_output;

    // 阈值0.5，
    faceDetector->detect(img, 0.5, scale, faceInfo);
    std::cout << "scale : " << scale << std::endl;
    cv::Mat face;

    cv::Mat src = img.clone();
    for (size_t i = 0; i < faceInfo.size(); i++)
    {
        if(faceInfo[i].rect.x1 < 0) faceInfo[i].rect.x1 = 0;
        if(faceInfo[i].rect.y1 < 0) faceInfo[i].rect.y1 = 0;
        if(faceInfo[i].rect.x2 * scale > img.cols) faceInfo[i].rect.x2 = img.cols/scale;
        if(faceInfo[i].rect.y2 * scale > img.rows) faceInfo[i].rect.y2 = img.rows/scale;

        cv::Rect rect = cv::Rect(cv::Point2f(faceInfo[i].rect.x1 * scale, faceInfo[i].rect.y1 * scale),
                                 cv::Point2f(faceInfo[i].rect.x2 * scale, faceInfo[i].rect.y2 * scale));

//        float expand_scale = 0.082;
//        int x1 = faceInfo[i].rect.x1 * scale, y1 = faceInfo[i].rect.y1 * scale, x2 = faceInfo[i].rect.x2 * scale, y2 = faceInfo[i].rect.y2 * scale;
//        int width = x2 - x1, height = y2 - y1;
//        int add_width = width * expand_scale, add_height = height * expand_scale;
//        x1 -= add_width/2;
//        y1 -= add_height/2;
//        x2 += add_width/2;
//        y2 += add_height/2;
//        if(x1 < 0) x1 = 0;
//        if(y1 < 0) y1 = 0;
//        if(x2 > img.cols) x2 = img.cols;
//        if(y2 > img.rows) y2 = img.rows;
//        cv::Rect rect = cv::Rect(x1, y1, width + add_width, height + add_height);
//
//        std::cout << "width : " << width << ", height : " << height << ", add_width : " << add_width << ", add_height : " << add_height << endl;
//        std::cout << "x1 : " << x1 << ", y1 : " << y1 << ", x2 : " << x2 << ", y2 : " << y2 << endl;
//        std::cout << "x : " << rect.x << ", y : " << rect.y << ", width : " << rect.width << ", height : " << rect.height << endl;

        face = src(rect);
        classifier->classify(face, classify_output, scale_classify);

        const string face_name = "../face/face_" + to_string(i) + ".jpg";
        imwrite(face_name, face);

        string text = "goggle_mask";
        string index = "0";
        float max = classify_output[0];
        for (int i = 1; i < 4; i++) {
            if(classify_output[i] > max) {
                max = classify_output[i];
                if(i == 1) {text = "goggle";index="1";}
                else if(i == 2) {text = "mask";index="2";}
                else if (i == 3) {text = "no";index="3";}
            }
        }

        std::cout << "classify result : " << text << std::endl;
        cv::putText(src, index, cv::Point(rect.x + rect.width/2,rect.y + rect.height/2), FONT_HERSHEY_SIMPLEX,1,Scalar(23,255,0),2,8);
        cv::rectangle(src, rect, Scalar(0, 0, 255), 2);

//        for (size_t j = 0; j < 5; j++)
//        {
//            cv::Point2f pt = cv::Point2f(faceInfo[i].pts.x[j] * scale, faceInfo[i].pts.y[j] * scale);
//            cv::circle(src, pt, 1, Scalar(0, 255, 0), 2);
//        }
    }

    std::cout << "detect face size : " << faceInfo.size() << std::endl;

    string result = "../result/";

    std::vector<string> paths = split(img_path, "/");
    result += paths[paths.size()-1];

    
    std::cout << "save result : " << result << endl;
    imwrite(result, src);

    // 测速
     float detect_time = 0, classify_time = 0;
     int count = 0;
     RK::Timer ti;

     while(1) {
         ti.reset();
         float scale =1.0;
         std::vector<FaceDetectInfo> faceInfo;
         faceDetector->detect(img, 0.5, scale, faceInfo);
         detect_time += ti.elapsedMilliSeconds();
         count ++;
         if(count % 1000 == 0) {
             printf("face detection average detect_time = %f.\n", detect_time / count);
         }

         ti.reset();
         classifier->classify(face, classify_output, scale_classify);
         classify_time += ti.elapsedMilliSeconds();
         if(count % 1000 == 0) {
             printf("face classify average classify_time = %f.\n", classify_time / count);
         }
     }


}

int main(int argc, char** argv) {

    if(argc != 3) {
        std::cout << "Help : \n please use commond \" ./goggleMask [classify|detect] [path/image.jpg | path/to/filelist.txt]\"\n";
        return 0;
    }

    string path = "../model";

    bool is_classify = false;


    if(strcmp("classify", argv[1]) == 0) {
        is_classify = true;
        std::cout << "classify ...\n";
    }

    RetinaFace *faceDetector = NULL;
    GoggleMaskClassify *classifier = NULL;

    if(!is_classify) {
        faceDetector = new RetinaFace(path, "net3");
    }

    classifier = new GoggleMaskClassify(path);


    string images = argv[2];
    if(!endsWith(images, ".txt")) {
        if (is_classify)
            classify(classifier, argv[2]);
        else
            detectFace(faceDetector, classifier, images.c_str());
    } else {

        ifstream in(images.c_str());
        string line;
        if (!in) {
            cout << "openfile : " << images << " fail ... \n";
            delete classifier;
            if (faceDetector != NULL) delete faceDetector;
            return 0;
        }
        int all = 0, correct = 0;
        while (getline(in, line)) {
            all++;
            if (is_classify)
                correct += classify(classifier, line.c_str());
            else
                detectFace(faceDetector, classifier, line.c_str());

        }
        cout << "all : " << all << ", correct : " << correct << ", pre : " << (float)correct/(float)all << endl;

    }

    delete classifier;
    if (faceDetector != NULL) delete faceDetector;

    return 0;
}

