// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <iostream>
#include <string>
#include <unordered_map>
#include <memory>

#include <ie_core.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include "openvino/openvino.hpp"

using namespace ov;


void run_model(std::shared_ptr<ov::Model> model) {
    ov::Core core;
    auto exeNetwork = core.compile_model(model, "CPU");
    auto infer_request = exeNetwork.create_infer_request();

    ov::Tensor input_tensor = infer_request.get_input_tensor();
    model->get_parameter_index(0);
    std::vector<float > fake_input(10, 1);
    std::memcpy(input_tensor.data<float>(), fake_input.data(), fake_input.size());

    infer_request.infer();
}


int main(int args, char *argv[]) {
    if (args < 4)
        return -1;

    ov::Core core;
    auto model = core.read_model(argv[1]);

    auto ordered_ops = model->get_ordered_ops();

    std::unordered_map<std::string, std::shared_ptr<Node>> name2op = {};

    //collect op mapps
    for(auto& op : ordered_ops) {
        name2op.emplace(op->get_friendly_name(), op);
    }
    std::vector<std::string> target_input = {argv[2]};
    std::vector<std::string> target_output = {argv[3]};

    std::vector<std::shared_ptr<opset8::Parameter> > subgraph_parameters = {};
    std::vector<std::shared_ptr<opset8::Result> > subgraph_results = {};
    for(auto& input_name : target_input) {
        auto input_op = name2op.at(input_name);
        if (auto node = ov::as_type_ptr<opset8::Parameter>(input_op)) {
            std::cout << "keep original parameter " << input_name << std::endl;
            subgraph_parameters.push_back(node);
            continue;
        }
        size_t num_const = 0;
        size_t index2non_const = -1;
        for(size_t i = 0; i < input_op->get_input_size(); i++) {
            auto parent = input_op->get_input_node_shared_ptr(i);
            if(ov::as_type_ptr<ov::opset8::Constant>(parent)) {
                num_const++;
            } else {
                if(index2non_const != -1) {
                    throw std::runtime_error("Too many none const inputs");
                } else {
                    index2non_const = i;
                }
            }
        }

        auto new_param = std::make_shared<opset8::Parameter>(input_op->get_input_element_type(index2non_const),
                                                             input_op->get_input_partial_shape(index2non_const));
        subgraph_parameters.push_back(new_param);
        input_op->input_value(index2non_const).replace(new_param->output(index2non_const));
    }

    for(auto& output_name : target_output) {
        auto output_op = name2op.at(output_name);
        if (output_op->get_output_size() !=1) {
            throw std::runtime_error("input must has 1 child");
        }
        auto node_copy = output_op->clone_with_new_inputs(output_op->input_values());
        auto new_result = std::make_shared<opset8::Result>(node_copy);
        ov::replace_node(output_op, new_result);
        subgraph_results.push_back(new_result);
    }

    auto subgraph = std::make_shared<ov::Model>(subgraph_results, subgraph_parameters);

    ov::pass::Serialize serializer("simple_bottom_mlp.xml", "simple_bottom_mlp.bin");
    serializer.run_on_model(subgraph);

}
