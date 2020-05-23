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
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel
"""Remote module"""

import tvm._ffi
from ... import relay
from ...rpc.client import RPCSession
from ...contrib import util


class RemoteModule(object):
    """Wrapper runtime module.

    This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output
    of underlying module functions

    Parameters
    ----------
    module : Module
        The internal tvm module that holds the actual remote functions.

    Attributes
    ----------
    module : Module
        The internal tvm module that holds the actual remote functions.
    """

    def __init__(self, module):
        self.module = module
        self.get_symbol = module["get_symbol"]

    def connect(self, url, port):
        fconnect = "tvm.remote_module.connect"
        m = tvm._ffi.get_global_func(fconnect)(self.module, url, port)
        self.remote = RPCSession(m);

    def build(self, mod, symbol, target, compile=True):
        name = None

        for var, func in mod.functions.items():
            if func.attrs and 'Compiler' in func.attrs and \
               func.attrs['Compiler'] == 'remote' and \
               str(var.name_hint) == symbol:
                name = str(var.name_hint)
                mod = tvm.IRModule.from_expr(func.body)
                break

        assert name is not None

        deploy_path = '/tmp/' + name  # TODO: should be customizable

        if compile:
            with relay.build_config(opt_level=3):
                json, lib, params = relay.build(mod, target=target)

            temp = util.tempdir()
            path = temp.relpath(name + '.tar')
            lib.export_library(path)
            self.remote.upload(path, deploy_path + '.tar')

            path = temp.relpath(name + '.json')
            with open(path, "w") as fo:
                fo.write(json)
            self.remote.upload(path, deploy_path + '.json')

            path = temp.relpath(name + '.params')
            with open(path, "wb") as fo:
                fo.write(relay.save_param_dict(params))
            self.remote.upload(path, deploy_path + '.params')

        ctx = self.remote.context(target)

        fload = "tvm.remote_module.load_graph"
        tvm._ffi.get_global_func(fload)(self.module, deploy_path, ctx)


@tvm._ffi.register_func("relay.ext.remote")
def create_remote_module(func):
    """
    Create a remote module from a Relay module.
    """
    assert isinstance(func, tvm.relay.function.Function)
    symbol = str(func.attrs.global_symbol)
    return tvm._ffi.get_global_func("tvm.remote_module.create")(symbol)
