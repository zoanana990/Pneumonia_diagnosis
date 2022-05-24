/******************************************************************************************
 * author @Hsin Yu Chen
 * Date @2021.12.22
 * file @dataset.h
 * Dataset Source: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
 ******************************************************************************************/
#ifndef FINAL_PROJECT_DATASET_H
#define FINAL_PROJECT_DATASET_H
#include <iostream>
#include <torch/torch.h>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
namespace fs = std::experimental::filesystem;

// declare a dataset which is inheritance the pytorch dataset
struct MedicalDataset: torch::data::datasets::Dataset<MedicalDataset>{
public:
    // We have two mode to do, first is training mode, and the second is testing mode
    enum Mode{kTrain, kVal, kTest};

    // in this structure we can see which mode we can choose
    explicit MedicalDataset(const std::string &root, const std::string& image_type,
                            Mode mode);

    // retrieve the example at an index
    torch::data::Example<> get(size_t index) override;

    // returns size of the dataset
    torch::optional<size_t> size() const override;

    // return Images and Labels
    const torch::Tensor& images() const;
    const torch::Tensor& targets() const;
private:

    torch::Tensor images_;
    torch::Tensor targets_;
    Mode mode_;
};
#endif //FINAL_PROJECT_DATASET_H