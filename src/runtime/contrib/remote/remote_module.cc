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
 * \file remote_module.cc
 */
#include "remote_module.h"

#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>

#include <sys/time.h>
#include <stdio.h>
#include <unistd.h>

namespace tvm {
namespace runtime {

PackedFunc RemoteModule::GetFunction(const std::string& name,
                                     const ObjectPtr<Object>& sptr_to_self) {
  if (name == "get_symbol") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->symbol_; });
  }
  if (name != symbol_) {
    return PackedFunc();
  }

  return PackedFunc([sptr_to_self, name, this](TVMArgs args, TVMRetValue* rv) {
    if (remote_arr_.size() == 0) {
      for (auto i = 0; i < args.size(); ++i) {
        CHECK(args[i].type_code() == kTVMDLTensorHandle) << "Expect DLTensor as inputs\n";
        DLTensor* local_in = args[i];
        std::vector<int64_t> shape(local_in->ndim);
        for (int i = 0; i < local_in->ndim; i++) {
          shape[i] = local_in->shape[i];
        }
        remote_arr_.push_back(NDArray::Empty(shape, local_in->dtype, ctx_));
      }
    }

    std::vector<TVMValue> tvm_values(args.size());
    std::vector<int> tvm_type_codes(args.size());
    TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());

    // Copy input tensors to corresponding data entries.
    for (auto i = 0; i < args.size(); ++i) {
      CHECK(args[i].type_code() == kTVMDLTensorHandle);
      if (i < args.size() - num_outputs_) {
        remote_arr_[i].CopyFrom(static_cast<DLTensor*>(args[i]));
      }
      setter(i, remote_arr_[i]);
    }

    frun_.CallPacked(TVMArgs(tvm_values.data(), tvm_type_codes.data(), args.num_args), rv);

    for (auto i = args.size() - num_outputs_; i < args.size(); ++i) {
      remote_arr_[i].CopyTo(static_cast<DLTensor*>(args[i]));
    }
    *rv = 0;
  });
}

void RemoteModule::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(url_);
  stream->Write(port_);
  stream->Write(symbol_);
  stream->Write(deploy_path_);

  TVMContext local_ctx = GetLocalContext(ctx_);
  stream->Write((int)local_ctx.device_type);
  stream->Write(local_ctx.device_id);
}

Module RemoteModule::Connect(const std::string& url, int port) {
  const PackedFunc* fconnect = runtime::Registry::Get("rpc.Connect");

  rmod_ = (*fconnect)(url, port, "");
  url_ = url;
  port_ = port;

  return rmod_;
}

TVMContext RemoteModule::GetLocalContext(TVMContext ctx) {
  return {
    .device_type = static_cast<DLDeviceType>((int)ctx.device_type % kRPCSessMask),
    .device_id = ctx.device_id,
  };
}

TVMContext RemoteModule::GetRemoteContext(TVMContext ctx) {
  const PackedFunc* findex = runtime::Registry::Get("rpc.SessTableIndex");
  int idx = (*findex)(rmod_);
  return {
    .device_type = static_cast<DLDeviceType>((int)ctx.device_type + kRPCSessMask * (idx + 1)),
    .device_id = ctx.device_id,
  };
}

void RemoteModule::LoadGraph(const std::string &deploy_path, const TVMContext &ctx) {
  PackedFunc fcreate = rmod_->GetFunction("tvm.wrap_graph_module.create");
  Module module = fcreate();

  PackedFunc fload = rmod_->GetFunction("tvm.wrap_graph_module.load_graph");
  fload(module, deploy_path, ctx);

  frun_ = module->GetFunction("run");
  num_outputs_ = module->GetFunction("get_num_outputs")();
  deploy_path_ = deploy_path;
  ctx_ = ctx;
}

Module RemoteModuleCreate(const std::string& symbol) {
  return Module(make_object<RemoteModule>(symbol));
}

void RemoteModuleLoadGraph(Module m, const std::string& deploy_path, TVMContext ctx) {
  RemoteModule* remote = static_cast<RemoteModule*>(m.operator->());

  remote->LoadGraph(deploy_path, ctx);
}

Module RemoteModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string url;
  int port;
  std::string symbol;
  std::string deploy_path;
  TVMContext local_ctx;

  CHECK(stream->Read(&url));
  CHECK(stream->Read(&port));
  CHECK(stream->Read(&symbol));
  CHECK(stream->Read(&deploy_path));
  CHECK(stream->Read(&local_ctx.device_type));
  CHECK(stream->Read(&local_ctx.device_id));

  auto exec = make_object<RemoteModule>(symbol);

  exec->Connect(url, port);
  TVMContext remote_ctx = exec->GetRemoteContext(local_ctx);
  exec->LoadGraph(deploy_path, remote_ctx);

  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.remote_module.create").set_body_typed(RemoteModuleCreate);

TVM_REGISTER_GLOBAL("tvm.remote_module.connect").set_body_typed([](Module m, const std::string& url, int port) {
  return static_cast<RemoteModule*>(m.operator->())->Connect(url, port);
});

TVM_REGISTER_GLOBAL("tvm.remote_module.load_graph").set_body_typed(RemoteModuleLoadGraph);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_remote").set_body_typed(RemoteModuleLoadBinary);

}  // namespace runtime
}  // namespace tvm
