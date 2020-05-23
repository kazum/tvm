/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file wrap_graph_runtime.cc
 */
#include "wrap_graph_runtime.h"
#include "../../file_util.h"

#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>

namespace tvm {
namespace runtime {

PackedFunc WrapGraphRuntime::GetFunction(const std::string& name,
                                      const ObjectPtr<Object>& sptr_to_self) {
  if (name == "get_num_outputs") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        int num_outputs = module_->GetFunction("get_num_outputs")();
        *rv = num_outputs;
      });
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {

        int num_outputs = module_->GetFunction("get_num_outputs")();
        int num_inputs = args.size() - num_outputs;
        // Copy input tensors to corresponding data entries.
        for (auto i = 0; i < num_inputs; ++i) {
          CHECK(args[i].type_code() == kTVMDLTensorHandle)
              << "Expect DLTensor as inputs\n";
          DLTensor* local_in = args[i];
          std::vector<int64_t> shape(local_in->ndim);
          for (int i = 0; i < local_in->ndim; i++) {
            shape[i] = local_in->shape[i];
          }
          NDArray remote_in = NDArray::Empty(shape, local_in->dtype, ctx_);
          remote_in.CopyFrom(local_in);
          module_->GetFunction("set_input")(i, remote_in);
        }

        module_->GetFunction("run")();

        for (auto i = 0; i < num_outputs; i++) {
          NDArray remote_out = module_->GetFunction("get_output")(i);
          int idx = num_inputs + i;
          CHECK(args[idx].type_code() == kTVMDLTensorHandle);
          remote_out.CopyTo(static_cast<DLTensor*>(args[idx]));
        }
        *rv = 0;
      });
  } else {
    return PackedFunc();
  }
}

void WrapGraphRuntime::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(deploy_path_);
  stream->Write((int)ctx_.device_type);
  stream->Write(ctx_.device_id);
}

void WrapGraphRuntime::LoadGraph(const std::string &deploy_path, const TVMContext &ctx) {
  const PackedFunc* fload = runtime::Registry::Get("tvm.rpc.server.load_module");
  Module m = (*fload)(deploy_path + ".tar");

  std::string json;
  LoadBinaryFromFile(deploy_path + ".json", &json);

  const PackedFunc* fcreate = runtime::Registry::Get("tvm.graph_runtime.create");
  module_ = (*fcreate)(json, m, (int)ctx.device_type, ctx.device_id);

  std::string params;
  LoadBinaryFromFile(deploy_path + ".params", &params);
  TVMByteArray arr = {
    .data = params.c_str(),
    .size = params.length(),
  };
  module_.GetFunction("load_params")(arr);

  deploy_path_ = deploy_path;
  ctx_ = ctx;
}

Module WrapGraphRuntimeCreate() {
  return Module(make_object<WrapGraphRuntime>());
}

void WrapGraphRuntimeLoadGraph(Module m, const std::string& deploy_path, TVMContext ctx) {
  WrapGraphRuntime* remote = static_cast<WrapGraphRuntime*>(m.operator->());

  remote->LoadGraph(deploy_path, ctx);
}

Module WrapGraphRuntimeLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string deploy_path;
  TVMContext ctx;

  CHECK(stream->Read(&deploy_path));
  CHECK(stream->Read(&ctx.device_type));
  CHECK(stream->Read(&ctx.device_id));

  auto exec = make_object<WrapGraphRuntime>();

  exec->LoadGraph(deploy_path, ctx);

  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.wrap_graph_module.create").set_body_typed(WrapGraphRuntimeCreate);

TVM_REGISTER_GLOBAL("tvm.wrap_graph_module.load_graph").set_body_typed(WrapGraphRuntimeLoadGraph);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_WrapGraphRuntime").set_body_typed(WrapGraphRuntimeLoadBinary);

}  // namespace runtime
}  // namespace tvm
