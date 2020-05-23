# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import tvm
from tvm import rpc, relay
from tvm.relay import transform
from tvm.contrib import util, xcode, graph_runtime
from tvm.contrib.target import coreml as _coreml
from tvm.contrib.target.remote import RemoteModule

import os
import re
import sys
import numpy as np

# Set to be address of tvm proxy.
proxy_host = os.environ["TVM_IOS_RPC_PROXY_HOST"]
# Set your desination via env variable.
# Should in format "platform=iOS,id=<the test device uuid>"
destination = os.environ["TVM_IOS_RPC_DESTINATION"]

if not re.match(r"^platform=.*,id=.*$", destination):
    print("Bad format: {}".format(destination))
    print("Example of expected string: platform=iOS,id=1234567890abcabcabcabc1234567890abcabcab")
    sys.exit(1)

proxy_port = 9090
key = "iphone"

# Change target configuration, this is setting for iphone6s
arch = "x86_64"
sdk = "iphonesimulator"
#arch = "arm64"
#sdk = "iphoneos"
target_host = "llvm -target=%s-apple-darwin" % arch
dtype = "float32"

def get_model(shape):
    x = relay.var('x', shape=shape)
    y = relay.var('data', shape=shape)
    _x = relay.annotation.compiler_begin(x, "remote")
    z = _x + _x
    z = relay.annotation.compiler_end(z, "remote")
    p = y * y
    f = relay.Function([x, y], z - p)

    x_data = np.random.rand(*shape).astype(dtype)
    mod = tvm.IRModule()
    mod["main"] = f
    mod = transform.PartitionGraph()(mod)

    return mod, {"x": x_data}


def test_mobilenet():
    temp = util.tempdir()
    shape = (10, 10)
    model, params = get_model(shape)

    def run(mod, target):
        with relay.build_config(opt_level=3):
            graph, lib, _params = relay.build(mod, target=target,
                                             target_host=target_host, params=params)
        print(graph)
        for m in lib.imported_modules:
            if m.type_key == "remote":
                remote = RemoteModule(m)
                symbol = remote.get_symbol()
                remote.connect(proxy_host, 9190)
                remote.build(mod, symbol, "llvm", True)

        path_dso = temp.relpath("deploy.dylib")
        lib.export_library(path_dso, xcode.create_dylib, arch=arch, sdk=sdk)

        xcode.codesign(path_dso)

        # Start RPC test server that contains the compiled library.
        xcode.popen_test_rpc(proxy_host, proxy_port, key,
                             destination=destination, libs=[path_dso])

        # connect to the proxy
        remote = rpc.connect(proxy_host, proxy_port, key=key)
        ctx = remote.cpu(0)
        lib = remote.load_module("deploy.dylib")
        m = graph_runtime.create(graph, lib, ctx)

        m.set_input('data', tvm.nd.array(np.random.uniform(size=shape).astype(dtype), ctx))
        m.set_input(**_params)
        m.run()
        m.get_output(0)

        # evaluate
        ftimer = m.module.time_evaluator("run", ctx, number=3, repeat=10)
        prof_res = np.array(ftimer().results) * 1000
        print("%-19s (%s)" % ("%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))

    def annotate(mod, compiler):
        """
        An annotator for Core ML.
        """
        # Bind free variables to the constant values.
        bind_dict = {}
        for arg in mod['main'].params:
            name = arg.name_hint
            if name in params:
                bind_dict[arg] = relay.const(params[name])

        mod['main'] = relay.bind(mod['main'], bind_dict)

        seq = tvm.transform.Sequential([
            transform.SimplifyInference(),
            transform.FoldConstant(),
            transform.FoldScaleAxis(),
            transform.AnnotateTarget(compiler),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph()
        ])

        with relay.build_config(opt_level=3):
            mod = seq(mod)

        print(mod)

        return mod

    run(annotate(model, "coremlcompiler"), target_host)

if __name__ == "__main__":
    test_mobilenet()
