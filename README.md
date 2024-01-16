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
ov_model_splitter <model path> <start end> <end node> [assign node]
```
# Notice
`start/end/assign` could be specified by configuration files, e.g.
```
input_1
input_2
```
```
output_1
output_2
```