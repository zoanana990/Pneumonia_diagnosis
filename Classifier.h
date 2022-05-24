/******************************************************************************************
 * author @Hsin Yu Chen
 * Date @2021.12.23
 * file @Classifier.h
 * Here I will declare a class which is used to train, predict, load model...
 ******************************************************************************************/
#ifndef FINAL_PROJECT_CLASSIFIER_H
#define FINAL_PROJECT_CLASSIFIER_H
#include "model.h"
#include "dataset.h"
#include <opencv2/opencv.hpp>
#include <torchvision/vision.h>
#include <torchvision/models/resnet.h>
class Classifier{
private:
    // torch::Device device must need to be initialized
    torch::Device device = torch::Device(torch::kCPU);

    // Resnet By me, it can also work, but need to modify some parameters
    ResNet r = resnet101(2);

    // torchvision model, I use this because
    // I compare the training speed between C++ and python
    vision::models::ResNet101 M = vision::models::ResNet101();
    torch::nn::Linear net = torch::nn::Linear(1000, 2);
    torch::nn::Sequential model = torch::nn::Sequential(M, net);

public:
    // Constructor
    explicit Classifier(int gpu=0);
    void Train(int num_epochs, int batch_size, float learning_rate,
               const std::string& train_val_dir, const std::string& image_type,
               const std::string& save_path);
    void Test();
    int Inference(cv::Mat &image, torch::Tensor& prediction);

    void LoadWeight(const std::string& weight_path);
};
#endif //FINAL_PROJECT_CLASSIFIER_H