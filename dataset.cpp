/******************************************************************************************
 * author @Hsin Yu Chen
 * github @zoanana990
 * Date @2021.12.22
 * file @dataset.cpp
 * Dataset Source: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
 ******************************************************************************************/
#include "dataset.h"
namespace{

    // the number of training dataset
    constexpr int kTrainSize = 2675; // number of images
    constexpr int kValSize = 470; // number of images
    constexpr int kTestSize = 16; // number of images
    constexpr int kRows = 512;
    constexpr int kCols = 512;

    torch::Tensor CVtoTensor(cv::Mat img){
        // First, we need to resize the image into our size
        cv::resize(img, img, cv::Size{kRows, kCols}, 0, 0, cv::INTER_LINEAR);

        // Second, we need to convert the image from BGR to RGB
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        // Third, copy the image from Mat to Torch Tensor
        auto img_tensor = torch::from_blob(img.data, {kRows, kCols, 3}, torch::kByte);

        // Forth, normalize the image into [0, 1]
        img_tensor = img_tensor.permute({2, 0, 1}).toType(torch::kFloat).div(255.0);

        return img_tensor;
    }
    std::pair<torch::Tensor, torch::Tensor> read_data(const std::string &root, const std::string& image_type,
                                                      MedicalDataset::Mode mode){


        // if the file is ends with ".jpg", ".jpeg", ".png"
        // notice the file end
        // TODO: std::vector<std::string> ext{".jpg", ".jpeg", ".png", ...};
//        std::string ext(image_type);

        // determine the data mode is, then we decide to go the file, and the dataset size
//        const auto num_samples = train ? kTrainSize : kTestSize;
//        const auto folder = train ? root + "/train" : root + "/test";

        int num_samples;
        std::string folder;
        if(mode == MedicalDataset::kTrain) num_samples=kTrainSize, folder=root+"/train";
        else if(mode == MedicalDataset::kVal) num_samples=kValSize, folder=root+"/val";
        else num_samples=kTestSize, folder=root+"/test";


        // this is the label for the image
        auto targets = torch::empty(num_samples, torch::kInt64);

        // this is the input of network
        // attention that the tensor size is {num_samples, 3, w, h} like the np.zeros((shape))
        auto images = torch::empty({num_samples, 3, kRows, kCols}, torch::kFloat);

        // we need to define the folder we will go
        std::string normal_folder = folder + "/NORMAL";
        std::string pneumontia_folder = folder + "/PNEUMONIA";
        std::vector<std::string> folders = {normal_folder, pneumontia_folder};
        int64_t label=0;
        // index
        int i = 0;
        // we need to dfs the folder system to fetch the data we need
        for(auto &f: folders){
            std::cout << label << std::endl;
            for(const auto &p : fs::directory_iterator(f)){
                //// Boundary Condition: "i" is the image index, and it cannot >= the number of samples
                if(i >= num_samples) break;

                // like python if string.endswith() == ...

                if(p.path().extension() == image_type){

//                    std::cout << "Start Reading " << p.path() << " File!" << " Which Index is " << i << std::endl;
                    std::cout << "Index: " << i << " | Filenames: " << p.path();
                    // read the image
                    cv::Mat img = cv::imread(p.path());

                    // convert the image to the torch tensor
                    auto img_tensor = CVtoTensor(img);
                    images[i] = img_tensor;
                    targets[i] = torch::tensor(label, torch::kInt64);
                    std::cout << " | Label is " << targets[i] << std::endl;
                    i++;
                }
            }
            label++;
        }
        std::cout << "Finish reading " << mode << " data" << std::endl;
        return {images, targets};
    }
} // namespace
MedicalDataset::MedicalDataset(const std::string &root, const std::string& image_type,
                               MedicalDataset::Mode mode): mode_(mode) {
    /* This is a constructor for the data, which is a pair of image and the target
     * */
    auto data = read_data(root, image_type, mode);
    images_ = std::move(data.first);
    targets_ = std::move(data.second);
}
torch::data::Example<> MedicalDataset::get(size_t index) {
    return {images_[index], targets_[index]};
}
torch::optional<size_t> MedicalDataset::size() const {
    return images_.size(0);
}
const torch::Tensor &MedicalDataset::images() const {
    return images_;
}
const torch::Tensor &MedicalDataset::targets() const {
    return targets_;
}