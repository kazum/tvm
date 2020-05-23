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
 * \brief Graph runtime which contains whole graph information.
 * \file wrap_graph_runtime.h
 */
#ifndef TVM_RUNTIME_CONTRIB_REMOTE_WRAP_GRAPH_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_REMOTE_WRAP_GRAPH_RUNTIME_H_

#include <dlpack/dlpack.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <memory>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

/*!
 * \brief Wrapped graph runtime.
 *
 *  This runtime can be accessed in various language via
 *  TVM runtime PackedFunc API.
 */
class WrapGraphRuntime : public ModuleNode {
 public:
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
  const char* type_key() const { return "WrapGraphRuntime"; }

  void LoadGraph(const std::string &deploy_path, const TVMContext &ctx);

 private:
  Module module_;
  std::string deploy_path_;
  TVMContext ctx_;
};

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_REMOTE_WRAP_GRAPH_RUNTIME_H_
