/******************************************************************************************
 * author @Hsin Yu Chen
 * Date @2021.12.23
 * file @Classifier.cpp
 * Here I will declare a class which is used to train, predict, load model...
 ******************************************************************************************/
#include "Classifier.h"
#define ImageWidth 512
#define ImageHeight 512
Classifier::Classifier(int gpu) {
    // if we can get the gpu
    device = gpu >= 0? torch::Device(torch::kCUDA): torch::Device(torch::kCPU);
}

void Classifier::Train(int num_epochs, int batch_size, float learning_rate, const std::string& train_val_dir,
                       const std::string& image_type, const std::string& save_path) {

    /**********************************************************************
     * Here I implement the training function
     *
     * In deep learning, there are some steps for training:
     *  1. fetch the data from dataset and convert them into torch::Tensor
     *  2. build the neural network model,
     *     and input the data to the neural network
     *  3. set the loss function, then compute the loss function
     *     between Ground Truth and the predicted value
     *  4. set the optimizer, which is used to reduce the loss by using
     *     some gradient method, here I choose SGD
     *  5. set hyperparameters, then start training
     *  6. save the training model
     **********************************************************************/

    const std::string& root = train_val_dir;

    // Read the training data, normalize and augment
    auto train_dataset = MedicalDataset(root, image_type, MedicalDataset::Mode::kTrain)
            .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
            .map(torch::data::transforms::Stack<>());

    // load the training data
    auto train_loader =
            torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset),
                                                                                batch_size);

    auto val_dataset = MedicalDataset(root, image_type, MedicalDataset::Mode::kVal)
            .map(torch::data::transforms::Normalize<>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5}))
            .map(torch::data::transforms::Stack<>());

    // load the testing data
    auto val_loader =
            torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(val_dataset),
                                                                                batch_size);
//    auto resnet101 = vision::models::ResNet101();
//    // here we need to add the output interface to use the net
//    auto net = torch::nn::Linear(1000, 2);
//    auto model = torch::nn::Sequential(resnet101, net);

    model->to(device); //activate gpu

    // Initialize the accuracy and the
    float loss_train=0.0, loss_test=0.0, accuracy_train=0.0, accuracy_test=0.0, best_accuracy=0.0;

    std::cout << "\nTraining Start" << std::endl;
    // Training Start
    for(size_t epoch=1; epoch<=num_epochs; epoch++){

        //// initialize batch index
        size_t batch_index_train=0, batch_index_val=0;

        //// adjust the learning rate at the half of training
        //// because at that time, loss is converged, so we need to reduce the learning rate at
        if(epoch == int(num_epochs / 2)) learning_rate/=10;

        //// set the optimizer
        torch::optim::SGD optimizer(model->parameters(),
                                    torch::optim::SGDOptions(learning_rate).momentum(0.9));

        for (auto mm : model->named_parameters())
            mm.value().set_requires_grad(true);

        for(auto& batch: *train_loader){
            auto data = batch.data.to(torch::kF32).to(device);
            auto target = batch.target.to(torch::kInt64).to(device).squeeze();
            //// squeeze is doing the dimension squeeze

            optimizer.zero_grad(); //// Reset the previous Gradient equal python

            // Execute the model
            torch::Tensor prediction = model->forward(data);
            prediction = prediction.view({prediction.size(0), -1});

            auto acc = prediction.argmax(1).eq(target).sum();

            //// acc is a torch::Tensor, acc.template item<float> is that convert the torch::Tensor to float
            //// take example: auto x = torch::Tensor([1.0])
            //// x.template item<float> -> x=1.0
            accuracy_train += acc.template item<float>() / (float)batch_size;

            //// compute loss value, here I use nll loss
            //// nll_loss is like cross entropy, but nll_loss is more convenient
            //// because nll_loss can do multi-classes classification
            torch::Tensor loss = torch::nll_loss(torch::log_softmax(prediction, 1), target);

            //// compute gradients
            loss.backward();

            //// do gradient algorithm and update the parameters in Neural Network
            //// this is equivalent to python
            optimizer.step();

            //// update loss
            loss_train += loss.item<float>();

            batch_index_train++;

            //// print
            std::cout << "Epoch: " << epoch << " | Batch Index Train: " << batch_index_train
            << " | Train Loss: " << loss_train / (float)batch_index_train
            << " | Train Accuracy: " << accuracy_train / (float)batch_index_train
            << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Start Validation" << std::endl;
        // validation parts
        model->eval();
        for(auto& batch: *val_loader){
            auto data = batch.data.to(device);
            auto target = batch.target.to(device).squeeze();

            torch::Tensor prediction = model->forward(data);

            // compute the loss
            torch::Tensor loss = torch::nll_loss(torch::log_softmax(prediction, 1), target);
//            torch::Tensor loss = torch::nll_loss(prediction, target);
//            std::cout << "loss.size = " << loss.sizes() << std::endl;

            auto acc = prediction.argmax(1).eq(target).sum();
            accuracy_test += acc.template item<float>() / (float)batch_size;

            loss_test += loss.item<float>();
            batch_index_val++;
            std::cout << "Epoch: " << epoch << " | Val Loss: " << loss_test / (float)batch_index_val
            << " | Valid Acc: " << accuracy_test / (float)batch_index_val << std::endl;
        }
        std::cout << std::endl;

        if(accuracy_test > best_accuracy){
            torch::save(model, save_path);
            best_accuracy = accuracy_test;
        }
        loss_train=0.0, loss_test=0.0, accuracy_train=0.0, accuracy_test=0.0;
    }
}
//// Do the Prediction here
int Classifier::Inference(cv::Mat &image, torch::Tensor& prediction) {

    // Read the data and resize
    cv::resize(image, image, cv::Size(ImageHeight, ImageWidth));

    // convert the image
    torch::Tensor img_tensor = torch::from_blob(image.data,
                                                {image.rows, image.cols, 3},
                                                torch::kByte).permute({2, 0, 1});

    img_tensor = img_tensor.to(device).unsqueeze(0).to(torch::kF32).div(255.0);

    // do the neural network prediction
    prediction = model->forward(img_tensor);
    prediction = torch::softmax(prediction, 1);

    // pick the maxima percentage
    auto class_id = prediction.argmax(1);
//    std::cout << "Prediction: " << prediction <<", Class ID is " << class_id << std::endl;

    int ans = int(class_id.item().toInt());

//    float prob = prediction[0][ans].item().toFloat();
    return ans;
}

void Classifier::LoadWeight(const std::string& weight_path) {
    torch::load(model, weight_path);
    model->eval();
}

void Classifier::Test() {
//TODO
}