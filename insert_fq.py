from cv2 import multiply, subtract
from openvino.runtime import Core, serialize, opset8, Type
from openvino.runtime.passes import ModelPass, Matcher, MatcherPass, WrapType, Manager, VisualizeTree, AnyInput, ConstantFolding
from openvino.runtime.utils import replace_node
import numpy as np
import sys
import json
core = Core()

class WeightQuantizationReplacement(MatcherPass):
    def __init__(self, weight_scales):
        MatcherPass.__init__(self)
        self.model_changed = False
        self.weight_scales = weight_scales
        weight = WrapType("opset8::Constant")
        convert = WrapType("opset8::Convert", weight.output(0))
        zero_point = WrapType("opset8::Constant")
        subtract = WrapType("opset8::Subtract", [convert.output(0), zero_point.output(0)])
        scales = WrapType("opset8::Constant")
        multiply = WrapType("opset8::Multiply", [subtract.output(0), scales.output(0)])

        def callback(m: Matcher) -> bool:
            self.applied = True
            weight_node = m.get_pattern_value_map()[weight].get_node()
            zero_point_node = m.get_pattern_value_map()[zero_point].get_node()
            new_zp = opset8.constant(0, zero_point_node.get_element_type(), zero_point_node.friendly_name)
            # For testing purpose
            self.model_changed = False
            replace_node(zero_point_node, new_zp)
            scale_node = m.get_pattern_value_map()[scales].get_node()
            new_scale = opset8.constant(np.expand_dims(np.array(self.weight_scales[weight_node.friendly_name]), axis=1), scale_node.get_element_type(), scale_node.friendly_name)
            replace_node(scale_node, new_scale)
            # self.register_new_node(new_relu)

            # Input->Relu->Result => Input->Relu->Relu->Result
            # root.input(0).replace_source_output(new_relu.output(0))
            return False

        self.register_matcher(Matcher(multiply, "WeightQuantizationReplacement"), callback)


class InsertQuantization(MatcherPass):
    def __init__(self, weight_scales):
        MatcherPass.__init__(self)
        self.model_changed = False
        self.weight_scales = weight_scales
        concat = WrapType("opset8::Concat", [AnyInput(), AnyInput()])
        weight = WrapType("opset8::Constant")
        convert = WrapType("opset8::Convert", weight.output(0))
        zero_point = WrapType("opset8::Constant")
        subtract = WrapType("opset8::Subtract", [convert.output(0), zero_point.output(0)])
        scales = WrapType("opset8::Constant")
        multiply = WrapType("opset8::Multiply", [subtract.output(0), scales.output(0)])
        matmul = WrapType("opset8::MatMul", [concat.output(0), multiply.output(0)])

        def callback(m: Matcher) -> bool:
            self.applied = True
            concat_node = m.get_pattern_value_map()[concat].get_node()
            matmul_node = m.get_pattern_value_map()[matmul].get_node()
            mutiply_node = m.get_pattern_value_map()[multiply].get_node()
            scales_node = m.get_pattern_value_map()[scales].get_node()
            zp_node = m.get_pattern_value_map()[zero_point].get_node()
            weight_node = m.get_pattern_value_map()[weight].get_node()
        
            const_scales = scales_node.get_vector()
            const_zp = zp_node.get_vector()
            const_weight = weight_node.get_vector()
            np.save("fc_scale", const_scales);
            np.save("fc_zp", const_zp);
            np.save("fc_weight", const_weight)
            input_low = opset8.constant(-5.12978, Type.f32, "input_low")
            input_high = opset8.constant(5.089652, Type.f32, "inpu_high")
            output_low = opset8.constant(-5.12978, Type.f32, "output_low")
            output_high = opset8.constant(5.089652, Type.f32, "output_high")
            new_fq = opset8.fake_quantize(concat_node, input_low, input_high, output_low, output_high, 256)
            new_matmul = opset8.matmul(new_fq, mutiply_node, False, True)
            # For testing purpose
            self.model_changed = False
            replace_node(matmul_node, new_matmul)
            # self.register_new_node(new_relu)

            # Input->Relu->Result => Input->Relu->Relu->Result
            # root.input(0).replace_source_output(new_relu.output(0))
            return False

        self.register_matcher(Matcher(matmul, "InsertQuantization"), callback)

class InsertQuantization2(MatcherPass):
    def __init__(self, weight_scales):
        MatcherPass.__init__(self)
        self.model_changed = False
        self.weight_scales = weight_scales
        concat = WrapType("opset8::Concat", [AnyInput(), AnyInput()])
        weight = WrapType("opset8::Constant")
        matmul = WrapType("opset8::MatMul", [concat.output(0), weight.output(0)])

        def callback(m: Matcher) -> bool:
            self.applied = True
            concat_node = m.get_pattern_value_map()[concat].get_node()
            matmul_node = m.get_pattern_value_map()[matmul].get_node()
            weight_node = m.get_pattern_value_map()[weight].get_node()
            input_low = opset8.constant(-5.12978, Type.f32, "input_low")
            input_high = opset8.constant(5.089652, Type.f32, "inpu_high")
            output_low = opset8.constant(-5.12978, Type.f32, "output_low")
            output_high = opset8.constant(5.089652, Type.f32, "output_high")
            const_scales = np.load("full_connected_scales.npy")
            const_zp = np.load("fc_zp.npy")
            const_weight = np.load("fc_weight.npy")
            scales2 = opset8.constant(const_scales.reshape(512, 1), Type.f32, "scales2")

            zp = opset8.constant(const_zp.reshape(512, 1), Type.f32, "zero_points")
            # div = opset8.divide(concat_node, scales);
            # convert2 = opset8.convert(convert1, "F32")
            # mul = opset8.multiply(convert2, scales2)
            new_weight = opset8.constant(const_weight.reshape(512, 415), Type.i8, "fake_weight")
            convert = opset8.convert(new_weight, "F32")
            sub = opset8.subtract(convert, zp)
            mul = opset8.multiply(sub, scales2)
            new_fq = opset8.fake_quantize(concat_node, input_low, input_high, output_low, output_high, 256)
            new_matmul = opset8.matmul(new_fq, mul, False, True)
            # For testing purpose
            self.model_changed = False
            replace_node(matmul_node, new_matmul)
            # self.register_new_node(new_relu)

            # Input->Relu->Result => Input->Relu->Relu->Result
            # root.input(0).replace_source_output(new_relu.output(0))
            return False

        self.register_matcher(Matcher(matmul, "InsertQuantization"), callback)

# model_path = "./bottom_mlp_int8/90_bottom_mlp_int8.xml"
# model_path = "/home/zhangyi7/ov_dlrm/results/dlrm_2048_10GB_int8_MinMaxQuantization/2022-05-20_21-41-16/optimized/dlrm_2048_10GB_int8.xml"
model_path = "/home/zhangyi7/ov_dlrm/results/dlrm_2048_10GB_int8_MinMaxQuantization/2022-06-11_12-51-12/optimized/dlrm_2048_10GB_int8.xml"
model = core.read_model(model_path)
ops = model.get_ordered_ops()
print(model.get_ordered_ops())

ops_to_modify = [
    {"Gemm_0/WithoutBiases/fq_input_0": 0},
    {"Gemm_2/WithoutBiases/fq_input_0": 2},
    {"Gemm_4/WithoutBiases/fq_input_0": 4}
]

weight_to_modify = [
    {"bot_l.0.weight2993579/quantized39156452", 0},
    {"bot_l.2.weight3043581/quantized40355795", 2},
    {"bot_l.4.weight3093583/quantized40656443", 4}
]

with open("int8_configure.json", "r") as f:
    int8_config = json.load(f)
print(len(int8_config[0]["weight_scales"][0]))
print(len(int8_config[2]["weight_scales"][0]))
print(len(int8_config[4]["weight_scales"][0]))
weight_scales = {
    "bot_l.0.weight3043676/quantized40836625": int8_config[0]["weight_scales"][0],
    "bot_l.2.weight3093678/quantized41736343": int8_config[2]["weight_scales"][0],
    "bot_l.4.weight3143680/quantized40236520": int8_config[4]["weight_scales"][0]
}
m = Manager()
# check that register pass returns pass instance
p = m.register_pass(InsertQuantization2(weight_scales))
# p = m.register_pass(ConstantFolding())
m.run_passes(model)
serialize(model, "dlrm_10_final.xml", "dlrm_10_final.bin")

