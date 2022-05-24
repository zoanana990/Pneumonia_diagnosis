/******************************************************************************************
 * author @Hsin Yu Chen
 * Date @2021.12.22
 * file @dataset.h
 * Resnet Paper: https://arxiv.org/pdf/1512.03385.pdf
 ******************************************************************************************/
#include "model.h"

// torch::nn::Conv2dOptions is repeated, so I make a common function
// additionally, the .stride(), .padding(), .bias() ... is a bad design, is too inconvenient to use
// pytorch version: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
inline torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
                                             int64_t stride = 1, int64_t padding = 0,
                                             int groups = 1, bool with_bias = true, int dilation = 1) {
    torch::nn::Conv2dOptions conv_options =
            torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride(stride); // stride number
    conv_options.padding(padding); // padding number
    conv_options.bias(with_bias);  // with bias or not, bias is wx+"b"
    conv_options.groups(groups); // controls the connection between inputs and outputs
    conv_options.dilation(dilation); // trous algorithm
    return conv_options;
}

BlockImpl::BlockImpl(int64_t inplanes, int64_t planes, int64_t Stride,
                     torch::nn::Sequential Downsample,
                     int groups, int base_width, bool Basic) {
    downsample = Downsample;
    stride = Stride;
    // output planes
    int width = int((double)planes * (base_width) / 64.) * groups;
//    std::cout << "width = " << width << std::endl;

    // basic block structure
    conv1 = torch::nn::Conv2d(conv_options(inplanes, width, 3, Stride,
                                           1, groups, false));
    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(width));
    conv2 = torch::nn::Conv2d(conv_options(inplanes, width, 3, 1,
                                           1, groups, false));
    bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(width));
    basic = Basic;
    // bottleneck structure if basic is false
    if(!basic){
        conv1 = torch::nn::Conv2d(conv_options(inplanes, width, 1,
                                               1, 0, 1, false));
//        bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(width));
        conv2 = torch::nn::Conv2d(conv_options(width, width, 3,
                                               Stride, 1, groups, false));
//        bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(width));
        conv3 = torch::nn::Conv2d(conv_options(width, 4 * planes, 1,
                                               1, 0, 1, false));
        bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(4 * planes));
    }

    // register block
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    if(!basic){
        register_module("conv3", conv3);
        register_module("bn3", bn3);
    }
    if (!downsample->is_empty()) {
        register_module("downsample", downsample);
    }
}

torch::Tensor BlockImpl::forward(torch::Tensor x) {
//    std::cout << "x.sizes()" << x.sizes() << std::endl;

//    std::cout << "\n Go 01" << std::endl;
    torch::Tensor residual = x.clone(); // do the identity mapping
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x); // activation function

//    std::cout << "\n Go 02" << std::endl;
    x = conv2->forward(x);
    x = bn2->forward(x);

//    std::cout << "\n Go 03" << std::endl;

    if(!basic){
        x = torch::relu(x);
        x = conv3->forward(x);
        x = bn3->forward(x);
    }
    if (!downsample->is_empty()) {
        residual = downsample->forward(residual);
    }
//    std::cout << "\n Go 04" << std::endl;

//    std::cout << "x.sizes() = " << x.sizes() << std::endl; // [4, 256, 128, 128]
//    std::cout << "residual.sizes() = " << residual.sizes() << std::endl; // [4, 64, 128, 128]

    x+=residual; // identity + convolution = residual
    x=torch::relu(x);

//    std::cout << "\n Go 05" << std::endl;

    return x;
}

ResNetImpl::ResNetImpl(std::vector<int> layers, int num_classes,
                       std::string ModelType, int Groups, int width_per_group) {
    /**********************************************************************
     * Here I divide Resnet model into two categories.
     * One is Resnet18 and Reset34, which are consists of basic block
     * others are consists of bottleneck block.
     *
     * these two categories is contolled by `basic`,
     **********************************************************************/
//     model_type = std::move(ModelType);
     // if use bottleneck
    if(ModelType != "resnet18" and ModelType != "resnet34"){
        expansion = 4;
        basic = false;
    }

    groups = Groups;
    base_width = width_per_group;

    // Assemble the same model
    conv1 = torch::nn::Conv2d(conv_options(3, 64, 7,
                                           2, 3, 1, false));
    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
    layer1 = torch::nn::Sequential(MakeLayer(64, layers[0])); /* stride = 1 */
    layer2 = torch::nn::Sequential(MakeLayer(128, layers[1], 2));
    layer3 = torch::nn::Sequential(MakeLayer(256, layers[2], 2));
    layer4 = torch::nn::Sequential(MakeLayer(512, layers[3], 2));
    fc = torch::nn::Linear(512*expansion, num_classes);

    // register the module
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("fc", fc);
}

torch::Tensor ResNetImpl::forward(torch::Tensor x) {
    /**********************************************************************
     * the forward function is same as resnet block
     * just do the forward propagation,
     * but this time, do not consider the residual structure
     **********************************************************************/
    std::cout << "\n Preparing" << std::endl;
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);

    // max_pool2d(self, kernel size, stride, padding)
    x = torch::max_pool2d(x, 3, 2, 1);

    // 4 layer
    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    // pool again, avg, not padding
    x = torch::avg_pool2d(x, 7, 1);
    std::cout << "x.avg_pool2d.sizes() = " << x.sizes() << std::endl; // [4, 2048, 10, 10]
    std::cout << "x.avg_pool2d.sizes()[0] = " << x.sizes()[0] << std::endl; // 4
    std::cout << "x.avg_pool2d.sizes()[1] = " << x.sizes()[1] << std::endl; // 2048
    std::cout << "x.avg_pool2d.sizes()[2] = " << x.sizes()[2] << std::endl; // 10

//    x = x.view({x.sizes()[0], -1});
    x = x.view({-1, x.sizes()[1]});
//    x = torch::nn::Linear(512 * expansion, classes)(x);
    std::cout << "x.view({x.sizes()[0], -1}) = " << x.sizes() << std::endl; // [4, 204800]

    // fully connection
    x = fc->forward(x); // ERROR: mat1 and mat2 shapes cannot be multiplied (4x204800 and 2048x2)
    std::cout << "x.fc(x) = " << x.sizes() << std::endl;

    // softmax
    x = torch::log_softmax(x, 1);
//    x = torch::nn::Linear(x,)
    std::cout << "torch::log_softmax(x, 1) = " << x.sizes() << std::endl;

    return x;
}

torch::nn::Sequential ResNetImpl::MakeLayer(int64_t planes, int64_t blocks, int64_t stride) {
    /**********************************************************************
     * I do the block assemble here, and make the same block as a layer
     * which is written in paper page 5, table 1
     **********************************************************************/
    torch::nn::Sequential downsample, layers;
    if(stride != 1 or inplanes != planes*expansion){
        downsample = torch::nn::Sequential(
                torch::nn::Conv2d(conv_options(inplanes, planes*expansion, 1,
                                               stride, 0, 1, false)),
                torch::nn::BatchNorm2d(planes*expansion)
                );
    }
    layers->push_back(Block(inplanes, planes, stride, downsample, groups, base_width, basic));
    inplanes = planes * expansion;
    for(int64_t i=1; i<blocks;i++)
        layers->push_back(Block(inplanes, planes, 1,
                                torch::nn::Sequential(), groups, base_width, basic));
    return layers;
}
// Build different Resnet
ResNet resnet18(int64_t num_classes){
    std::vector<int> layers={2, 2, 2, 2};
    ResNet model(layers, num_classes, "resnet18");
    return model;
}
ResNet resnet34(int64_t num_classes){
    std::vector<int> layers={3, 4, 6, 3};
    ResNet model(layers, num_classes, "resnet34");
    return model;
}
ResNet resnet50(int64_t num_classes){
    std::vector<int> layers={3, 4, 6, 3};
    ResNet model(layers, num_classes, "resnet50");
    return model;
}
ResNet resnet101(int64_t num_classes){
    std::vector<int> layers={3, 4, 23, 3};
    ResNet model(layers, num_classes, "resnet101");
    return model;
}
ResNet resnet152(int64_t num_classes){
    std::vector<int> layers={3, 8, 36, 3};
    ResNet model(layers, num_classes, "resnet152");
    return model;
}