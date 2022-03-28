#include <iostream>
#include <iostream>
#include <ie_core.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include "openvino/openvino.hpp"

using namespace ov;

std::shared_ptr<ov::Model>  getConvMat() {
    auto input = std::make_shared<ov::opset8::Parameter>(element::f32, Shape{1, 1, 97, 770});
    auto kernel = std::make_shared<opset8::Constant>(element::f32, Shape{1, 1, 3, 3}, 1.0);
    auto inputLowNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, 0.0);
    auto inputHighNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, 255.0);
    auto outputLowNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, 0.0);
    auto outputHighNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, 255.0);
    auto filterInputLowNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, -128.0);
    auto filterInputHighNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, 127.0);
    auto filterOutputLowNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, -128.0);
    auto filterOutputHighNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, 127.0);
    auto mat_weight = std::make_shared<opset8::Constant>(element::f32, Shape{768, 768}, 0.5);
    auto output_bias = std::make_shared<opset8::Constant>(element::bf16, Shape{}, 1);
    auto output_convert = std::make_shared<opset8::Convert>(output_bias, element::f32);

    auto fq = std::make_shared<opset8::FakeQuantize>(input, inputLowNode, inputHighNode, outputLowNode, outputHighNode, 256);
    auto convWeightsFQNode = std::make_shared<opset8::FakeQuantize>(kernel,
                                                                    filterInputLowNode, filterInputHighNode, filterOutputLowNode, filterOutputHighNode, 256);
    auto conv = std::make_shared<opset8::Convolution>(fq, convWeightsFQNode, Strides{1, 1}, CoordinateDiff{0, 0}, CoordinateDiff{0, 0}, Strides{1, 1});

    auto matWeightsFQNode = std::make_shared<opset8::FakeQuantize>(mat_weight,
                                                                   filterInputLowNode, filterInputHighNode, filterOutputLowNode, filterOutputHighNode, 256);

    auto convFQNode = std::make_shared<opset8::FakeQuantize>(conv,
                                                             filterInputLowNode, filterInputHighNode, filterOutputLowNode, filterOutputHighNode, 256);
    auto axesNode = std::make_shared<opset8::Constant>(element::f32, Shape{}, 1);
    auto squeezeNode = std::make_shared<opset8::Squeeze>(convFQNode, axesNode);
    auto fc = std::make_shared<opset8::MatMul>(squeezeNode, matWeightsFQNode, false, true);
    auto result_full = std::make_shared<opset8::Result>(fc->output(0));
    auto model = std::make_shared<ov::Model>(result_full, ParameterVector{input}, "matNet");
    input->output(0).set_names({"input"});
    result_full->output(0).set_names({"Result"});
    return model;
};

std::shared_ptr<ov::Model> getFullConnected() {
    auto input = std::make_shared<ov::opset8::Parameter>(element::f32, Shape{1, 95, 768});

    auto inputLowNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, -127.0);
    auto inputHighNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, 128.0);
    auto outputLowNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, -127.0);
    auto outputHighNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, 128.0);

    auto matInputLowNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, -128.0);
    auto matInputHighNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, 127.0);
    auto matOutputLowNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, -128.0);
    auto matOutputHighNode = std::make_shared<opset8::Constant>(element::f32, Shape{1}, 127.0);

    auto mat_weight = std::make_shared<opset8::Constant>(element::f32, Shape{768, 768}, 1);
    auto output_bias = std::make_shared<opset8::Constant>(element::bf16, Shape{}, 1);
    auto output_convert = std::make_shared<opset8::Convert>(output_bias, element::f32);

    auto fq = std::make_shared<opset8::FakeQuantize>(input, matInputLowNode, matInputHighNode, matOutputLowNode, matOutputHighNode, 256);

    auto matWeightsFQNode = std::make_shared<opset8::FakeQuantize>(mat_weight,
                                                                   matInputLowNode, matInputHighNode, matOutputLowNode, matOutputHighNode, 256);


    auto fc = std::make_shared<opset8::MatMul>(fq, matWeightsFQNode, false, true);
    auto add = std::make_shared<opset8::Add>(fc, output_convert);
    auto output_fake = std::make_shared<opset8::FakeQuantize>(add, inputLowNode, inputHighNode, outputLowNode, outputHighNode, 256);
    auto result_full = std::make_shared<opset8::Result>(output_fake->output(0));
    auto model = std::make_shared<ov::Model>(result_full, ParameterVector{input}, "matNet");
    input->output(0).set_names({"input"});
    result_full->output(0).set_names({"Result"});
    return model;
};

int main() {


    auto model = getFullConnected();


    std::vector<float> fakeImg(1*95*768, 1);


    ov::pass::Serialize serializer("simple_matmul.xml", "simple_matmul.bin");
    serializer.run_on_model(std::const_pointer_cast<ov::Model>(model));

    using namespace InferenceEngine;
    ov::Core core;
    auto exeNetwork = core.compile_model(model, "CPU");
    auto infer_request = exeNetwork.create_infer_request();

    ov::Tensor input_tensor = infer_request.get_input_tensor();
    std::memcpy(input_tensor.data<float>(), fakeImg.data(), fakeImg.size());

    infer_request.infer();

    auto tensor = infer_request.get_output_tensor();
    auto outputShape = tensor.get_shape();
    float* out = tensor.data<float>();

    for(auto i = 0; i < outputShape[0]; i++) {
        for(auto j = 0; j < outputShape[1]; j++) {
            for(auto z = 0; z < outputShape[2]; z++) {
                auto index = i * outputShape[1] * outputShape[2] + j * outputShape[2] + z;
                std::cout << index << "|" << out[index] << " ";
            }
            std::cout << std::endl;
        }
    }


}
