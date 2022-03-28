#include <iostream>
#include <iostream>
#include <ie_core.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include "openvino/openvino.hpp"

using namespace ov;
int main() {
    auto image = std::make_shared<ov::opset8::Parameter>(element::f32, Shape{1, 4,224,224});
    auto kernel = std::make_shared<opset8::Constant>(element::f32, Shape{1, 4, 3, 3}, 1.0);
    auto inputLowNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, 0.0);
    auto inputHighNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, 255.0);
    auto outputLowNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, 0.0);
    auto outputHighNode = std::make_shared<opset8::Constant>(element::f32, Shape{1, 4, 224, 224}, 255.0);
    auto filterInputLowNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, -128.0);
    auto filterInputHighNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, 127.0);
    auto filterOutputLowNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, -128.0);
    auto filterOutputHighNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, 127.0);

    auto fq = std::make_shared<opset8::FakeQuantize>(image, inputLowNode, inputHighNode, outputLowNode, outputHighNode, 256);
    auto convWeightsFQNode = std::make_shared<opset8::FakeQuantize>(kernel,
                                                                    filterInputLowNode, filterInputHighNode, filterOutputLowNode, filterOutputHighNode, 256);
    auto conv = std::make_shared<opset8::Convolution>(fq, convWeightsFQNode, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});
//    auto reshape_2 = std::make_shared<op::v1::Reshape>(image, new_shape, false);
//    auto pool2 = std::make_shared<op::v1::AvgPool>(reshape_1, Strides{1,1}, Shape{0,0}, Shape{0,0}, Shape{3, 3}, false);
//
//    auto concat = std::make_shared<opset8::Concat>(OutputVector{pool1, pool2}, 1);
    auto result_full = std::make_shared<opset8::Result>(conv->output(0));
    auto model = std::make_shared<ov::Model>(result_full, ParameterVector{image}, "simpleNet");
    image->output(0).set_names({"Image"});
    result_full->output(0).set_names({"Result"});
//    image->get_output_element_type(0);
    // apply preprocessing
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
//
    // 1) InputInfo() with no args assumes a model has a single input
    ov::preprocess::InputInfo& input_info = ppp.input();
    input_info.tensor().set_layout("NCHW").set_element_type(element::f32);
    // 3) Here we suppose model has 'NCHW' layout for input
    input_info.model().set_layout("NCHW");
//
//    // 4) Once the build() method is called, the preprocessing steps
//    // for layout and precision conversions are inserted automatically
//    model = ppp.build();

    std::vector<float> fakeImg(1*3*224*224);
    std::iota(fakeImg.begin(), fakeImg.end(), 0);


    ov::pass::Serialize serializer("new_simple.xml", "new_simple.bin");
    serializer.run_on_model(std::const_pointer_cast<ov::Model>(model));
//    CNNNetwork network(model);
//    network.serialize("simple.xml");
    using namespace InferenceEngine;
    ov::Core core;
//    auto model = core.read_model("/home/zhangyi7/Downloads/resnet50-v2-7.onnx");
    auto exeNetwork = core.compile_model(model, "CPU");
    auto infer_request = exeNetwork.create_infer_request();

    ov::Tensor input_tensor = infer_request.get_input_tensor();
    std::memcpy(input_tensor.data<float>(), fakeImg.data(), fakeImg.size());

    for(int i = 0; i < 10; i++) {
        infer_request.infer();
    }
}
