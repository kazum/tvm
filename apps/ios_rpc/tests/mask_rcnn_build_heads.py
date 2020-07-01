import mask_rcnn
from tvm import relay
import tvm
from tvm.contrib import graph_runtime
import numpy as np
import os

shape = (1, 16, 40, 30)
target = 'cuda -libs=cublas,cudnn'
ctx = tvm.gpu()
#target = 'llvm'
#ctx = tvm.cpu()

param_path = "/".join([os.path.dirname(__file__), "mask_rcnn.params"])
loaded_params = relay.load_param_dict(bytearray(open(param_path, "rb").read()))
params = {}
for k, v in loaded_params.items():
    params[int(k)] = relay.const(v)

features = relay.var("features", shape=shape, dtype='float16')
boxes, labels, scores, masks = mask_rcnn.graph_heads(features, params)
net = relay.Tuple((boxes, labels, scores, masks))
mod = tvm.IRModule()
mod["main"] = relay.Function(relay.analysis.free_vars(net), net)

print(mod)

with relay.build_config(opt_level=0):
    graph, lib, params = relay.build(mod, target)

name = 'remote_295'

lib.export_library(name + ".tar")
with open(name + ".json", "w") as fo:
    fo.write(graph)
with open(name + ".params", "wb") as fo:
    fo.write(relay.save_param_dict(params))

m = graph_runtime.create(graph, lib, ctx)
m.set_input(**params)
m.run()

# evaluate
ftimer = m.module.time_evaluator("run", ctx, number=1, repeat=10)
prof_res = np.array(ftimer().results) * 1000  # multiply 1000 for converting to millisecond
print("%-19s (%s)" % ("%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))
