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
 * \brief Remote module that can run remote model
 *        containing only tvm PackedFunc.
 * \file remote_module.h
 */
#ifndef TVM_RUNTIME_CONTRIB_REMOTE_REMOTE_MODULE_H_
#define TVM_RUNTIME_CONTRIB_REMOTE_REMOTE_MODULE_H_

#include <dlpack/dlpack.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <memory>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief Remote module.
 *
 *  This runtime can be accessed in various language via
 *  TVM runtime PackedFunc API.
 */
class RemoteModule : public ModuleNode {
 public:
  explicit RemoteModule(const std::string& symbol) : symbol_(symbol) {}
  /*!
   * \brief Get member function to front-end.
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  virtual PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self);

  /*!
   * \brief Serialize the content of the mlmodelc directory and save it to
   *        binary stream.
   * \param stream The binary stream to save to.
   */
  void SaveToBinary(dmlc::Stream* stream) final;

  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const { return "remote"; }

  Module Connect(const std::string& url, int port);
  TVMContext GetLocalContext(TVMContext ctx);
  TVMContext GetRemoteContext(TVMContext ctx);
  void LoadGraph(const std::string &deploy_path, const TVMContext &ctx);

 private:
  std::string url_;
  int port_;
  std::string symbol_;
  std::string deploy_path_;
  TVMContext ctx_;
  std::vector<NDArray> remote_arr_;
  Module rmod_;
  PackedFunc frun_;
  int num_outputs_;
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_REMOTE_REMOTE_MODULE_H_
