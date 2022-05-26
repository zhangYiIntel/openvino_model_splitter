# OpenVINO Model Spillter

This tool provides a subgraph extractor to extract a subgraph from OpenVINO IR models.

# Build
make sure you have openvino installed and you have source the `setupvars.sh`, otherwise you need to set `-DInferenceEngine_DIR=<your ov build dir>` for cmake
```
mkdir build
cmake ..
make
```
# Usage

```bash
ov_model_splitter <model path> <start end> <end node>
```
# Limitation
Currently, this tool only supports single input/output from command line, but you can specify multiple inputs/outputs inside the source code at L47-L48
```
    std::vector<std::string> target_input = {argv[2]};
    std::vector<std::string> target_output = {argv[3]};
```