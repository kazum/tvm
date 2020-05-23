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

if(NOT USE_REMOTE_RUNTIME STREQUAL "OFF")
  message(STATUS "Build with contrib.remote")
  file(GLOB REMOTE_CONTRIB_SRC src/runtime/contrib/remote/*.cc)
  list(APPEND RUNTIME_SRCS ${REMOTE_CONTRIB_SRC})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${REMOTE_CONTRIB_LIB})
endif()
