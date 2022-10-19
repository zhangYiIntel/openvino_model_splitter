// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <iostream>
#include <string>
#include <unordered_map>
#include <memory>
#include <fstream>

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
    std::string inputs(argv[2]);
    std::string outputs(argv[3]);
    bool hasInputConfig = false;
    bool hasOutputConfig = false;
    if (inputs.find(".config") != std::string::npos) {
        hasInputConfig = true;
    }
    if (outputs.find(".config") != std::string::npos) {
        hasOutputConfig = true;
    }
    ov::Core core;
    auto model = core.read_model(argv[1]);

    auto ordered_ops = model->get_ordered_ops();

    std::unordered_map<std::string, std::shared_ptr<Node>> name2op = {};

    //collect op mapps
    for(auto& op : ordered_ops) {
        name2op.emplace(op->get_friendly_name(), op);
    }
    auto readFile = [](std::string& name) {
        std::cout << "Read Config " << name << std::endl;
        std::fstream newfile;
        std::vector<std::string> names;
        newfile.open(name, std::ios::in); 
        if (newfile.is_open()){   
            std::string tp;
            while(getline(newfile, tp)){ 
                names.push_back(tp);
            }
            newfile.close(); //close the file object.
        }
        return names;
    };

    std::vector<std::string> target_input = hasInputConfig ? readFile(inputs) : std::vector<std::string>{inputs}; // "offset", "att_cache", "cnn_cache"};
    std::vector<std::string> target_output = hasOutputConfig ? readFile(outputs) : std::vector<std::string>{outputs};
    std::cout << "Start " << target_input[0] << std::endl;
    std::cout << "End " << target_output[0] << std::endl;
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
        std::vector<int> index2non_const = {};
        for(size_t i = 0; i < input_op->get_input_size(); i++) {
            auto parent = input_op->get_input_node_shared_ptr(i);
            if(ov::as_type_ptr<ov::opset8::Constant>(parent)) {
                num_const++;
            } else {
               index2non_const.push_back(i);
            }
        }
        for(auto& index : index2non_const) {
            auto new_param = std::make_shared<opset8::Parameter>(input_op->get_input_element_type(index),
                input_op->get_input_partial_shape(index));
                input_op->input_value(index).replace(new_param->output(0));
                subgraph_parameters.push_back(new_param);
        }       
    }

    for(auto& output_name : target_output) {
        auto output_op = name2op.at(output_name);
        if (auto node = ov::as_type_ptr<opset8::Result>(output_op)) {
            std::cout << "keep original result " << output_name << std::endl;
            subgraph_results.push_back(node);
            continue;
        }

        if (output_op->get_output_size() !=1) {
            throw std::runtime_error("input must has 1 child");
        }
        auto node_copy = output_op->clone_with_new_inputs(output_op->input_values());
        auto new_result = std::make_shared<opset8::Result>(node_copy);
        ov::replace_node(output_op, new_result);
        subgraph_results.push_back(new_result);
    }

    auto subgraph = std::make_shared<ov::Model>(subgraph_results, subgraph_parameters);

    ov::pass::Serialize serializer("simple_model.xml", "simple_model.bin");
    serializer.run_on_model(subgraph);

}
