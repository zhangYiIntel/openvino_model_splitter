from openvino.runtime import Core
from openvino.runtime.passes import VisualizeTree
import sys
import os
model_path = sys.argv[1]

from openvino.runtime.passes import Manager
model_path = sys.argv[1]
print("Going to process {0}".format(model_path))
core = Core()
res_func = core.read_model(model=model_path)
model_name = os.path.basename(model_path).split('.')[0]
xml_path = model_name +"_cf.xml"
bin_path = model_name +"_cf.bin"
pass_manager = Manager()

pass_manager.register_pass("ConstantFolding")
pass_manager.register_pass("Serialize", xml_path, bin_path)
pass_manager.register_pass(VisualizeTree("visualize.svg"))
pass_manager.run_passes(res_func)

