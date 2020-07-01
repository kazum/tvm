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
import tvm.relay as relay
from tvm.contrib.target.remote import RemoteModule
from tvm.contrib import graph_runtime
from tvm.relay.expr_functor import ExprMutator
from tvm.relay.op.annotation import compiler_begin, compiler_end
import numpy as np
import os


inff = 3e+38

def graph_backbone(image, params):
    v0 = relay.expand_dims(image, axis=0) # /* ty=Tensor[(1, 3, 640, 480), float32] */;
    v1 = relay.nn.conv2d(v0, params[2], strides=[2, 2], padding=[1, 1, 1, 1], kernel_size=[3, 3]) # /* ty=Tensor[(1, 32, 320, 240), float32] */;
    v2 = relay.nn.batch_norm(v1, params[4], params[5], params[6], params[7]) # /* ty=(Tensor[(1, 32, 320, 240), float32], Tensor[(32), float32], Tensor[(32), float32]) */;
    v3 = v2[0];
    v4 = relay.nn.relu(v3) # /* ty=Tensor[(1, 32, 320, 240), float32] */;
    v5 = relay.nn.conv2d(v4, params[10], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 32, 320, 240), float32] */;
    v6 = relay.nn.batch_norm(v5, params[12], params[13], params[14], params[15]) # /* ty=(Tensor[(1, 32, 320, 240), float32], Tensor[(32), float32], Tensor[(32), float32]) */;
    v7 = v6[0];
    v8 = relay.nn.relu(v7) # /* ty=Tensor[(1, 32, 320, 240), float32] */;
    v9 = relay.nn.conv2d(v8, params[18], padding=[1, 1, 1, 1], groups=32, kernel_size=[3, 3]) # /* ty=Tensor[(1, 32, 320, 240), float32] */;
    v10 = relay.nn.conv2d(v9, params[20], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 16, 320, 240), float32] */;
    v11 = relay.nn.batch_norm(v10, params[22], params[23], params[24], params[25]) # /* ty=(Tensor[(1, 16, 320, 240), float32], Tensor[(16), float32], Tensor[(16), float32]) */;
    v12 = v11[0];
    v13 = relay.nn.conv2d(v12, params[27], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 96, 320, 240), float32] */;
    v14 = relay.nn.batch_norm(v13, params[29], params[30], params[31], params[32]) # /* ty=(Tensor[(1, 96, 320, 240), float32], Tensor[(96), float32], Tensor[(96), float32]) */;
    v15 = v14[0];
    v16 = relay.nn.relu(v15) # /* ty=Tensor[(1, 96, 320, 240), float32] */;
    v17 = relay.nn.conv2d(v16, params[35], strides=[2, 2], padding=[1, 1, 1, 1], groups=96, kernel_size=[3, 3]) # /* ty=Tensor[(1, 96, 160, 120), float32] */;
    v18 = relay.nn.conv2d(v17, params[37], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 24, 160, 120), float32] */;
    v19 = relay.nn.batch_norm(v18, params[39], params[40], params[41], params[42]) # /* ty=(Tensor[(1, 24, 160, 120), float32], Tensor[(24), float32], Tensor[(24), float32]) */;
    v20 = v19[0];
    v21 = relay.nn.conv2d(v20, params[44], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 144, 160, 120), float32] */;
    v22 = relay.nn.batch_norm(v21, params[46], params[47], params[48], params[49]) # /* ty=(Tensor[(1, 144, 160, 120), float32], Tensor[(144), float32], Tensor[(144), float32]) */;
    v23 = v22[0];
    v24 = relay.nn.relu(v23) # /* ty=Tensor[(1, 144, 160, 120), float32] */;
    v25 = relay.nn.conv2d(v24, params[52], padding=[1, 1, 1, 1], groups=144, kernel_size=[3, 3]) # /* ty=Tensor[(1, 144, 160, 120), float32] */;
    v26 = relay.nn.conv2d(v25, params[54], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 24, 160, 120), float32] */;
    v27 = relay.nn.batch_norm(v26, params[56], params[57], params[58], params[59]) # /* ty=(Tensor[(1, 24, 160, 120), float32], Tensor[(24), float32], Tensor[(24), float32]) */;
    v28 = v27[0];
    v29 = relay.add(v28, v20) # /* ty=Tensor[(1, 24, 160, 120), float32] */;
    v30 = relay.nn.conv2d(v29, params[62], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 144, 160, 120), float32] */;
    v31 = relay.nn.batch_norm(v30, params[64], params[65], params[66], params[67]) # /* ty=(Tensor[(1, 144, 160, 120), float32], Tensor[(144), float32], Tensor[(144), float32]) */;
    v32 = v31[0];
    v33 = relay.nn.relu(v32) # /* ty=Tensor[(1, 144, 160, 120), float32] */;
    v34 = relay.nn.conv2d(v33, params[70], strides=[2, 2], padding=[1, 1, 1, 1], groups=144, kernel_size=[3, 3]) # /* ty=Tensor[(1, 144, 80, 60), float32] */;
    v35 = relay.nn.conv2d(v34, params[72], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 32, 80, 60), float32] */;
    v36 = relay.nn.batch_norm(v35, params[74], params[75], params[76], params[77]) # /* ty=(Tensor[(1, 32, 80, 60), float32], Tensor[(32), float32], Tensor[(32), float32]) */;
    v37 = v36[0];
    v38 = relay.nn.conv2d(v37, params[79], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 192, 80, 60), float32] */;
    v39 = relay.nn.batch_norm(v38, params[81], params[82], params[83], params[84]) # /* ty=(Tensor[(1, 192, 80, 60), float32], Tensor[(192), float32], Tensor[(192), float32]) */;
    v40 = v39[0];
    v41 = relay.nn.relu(v40) # /* ty=Tensor[(1, 192, 80, 60), float32] */;
    v42 = relay.nn.conv2d(v41, params[87], padding=[1, 1, 1, 1], groups=192, kernel_size=[3, 3]) # /* ty=Tensor[(1, 192, 80, 60), float32] */;
    v43 = relay.nn.conv2d(v42, params[89], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 32, 80, 60), float32] */;
    v44 = relay.nn.batch_norm(v43, params[91], params[92], params[93], params[94]) # /* ty=(Tensor[(1, 32, 80, 60), float32], Tensor[(32), float32], Tensor[(32), float32]) */;
    v45 = v44[0];
    v46 = relay.add(v45, v37) # /* ty=Tensor[(1, 32, 80, 60), float32] */;
    v47 = relay.nn.conv2d(v46, params[97], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 192, 80, 60), float32] */;
    v48 = relay.nn.batch_norm(v47, params[99], params[100], params[101], params[102]) # /* ty=(Tensor[(1, 192, 80, 60), float32], Tensor[(192), float32], Tensor[(192), float32]) */;
    v49 = v48[0];
    v50 = relay.nn.relu(v49) # /* ty=Tensor[(1, 192, 80, 60), float32] */;
    v51 = relay.nn.conv2d(v50, params[105], padding=[1, 1, 1, 1], groups=192, kernel_size=[3, 3]) # /* ty=Tensor[(1, 192, 80, 60), float32] */;
    v52 = relay.nn.conv2d(v51, params[107], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 32, 80, 60), float32] */;
    v53 = relay.nn.batch_norm(v52, params[109], params[110], params[111], params[112]) # /* ty=(Tensor[(1, 32, 80, 60), float32], Tensor[(32), float32], Tensor[(32), float32]) */;
    v54 = v53[0];
    v55 = relay.add(v54, v46) # /* ty=Tensor[(1, 32, 80, 60), float32] */;
    v56 = relay.nn.conv2d(v55, params[115], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 192, 80, 60), float32] */;
    v57 = relay.nn.batch_norm(v56, params[117], params[118], params[119], params[120]) # /* ty=(Tensor[(1, 192, 80, 60), float32], Tensor[(192), float32], Tensor[(192), float32]) */;
    v58 = v57[0];
    v59 = relay.nn.relu(v58) # /* ty=Tensor[(1, 192, 80, 60), float32] */;
    v60 = relay.nn.conv2d(v59, params[123], strides=[2, 2], padding=[1, 1, 1, 1], groups=192, kernel_size=[3, 3]) # /* ty=Tensor[(1, 192, 40, 30), float32] */;
    v61 = relay.nn.conv2d(v60, params[125], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 64, 40, 30), float32] */;
    v62 = relay.nn.batch_norm(v61, params[127], params[128], params[129], params[130]) # /* ty=(Tensor[(1, 64, 40, 30), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
    v63 = v62[0];
    v64 = relay.nn.conv2d(v63, params[132], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 384, 40, 30), float32] */;
    v65 = relay.nn.batch_norm(v64, params[134], params[135], params[136], params[137]) # /* ty=(Tensor[(1, 384, 40, 30), float32], Tensor[(384), float32], Tensor[(384), float32]) */;
    v66 = v65[0];
    v67 = relay.nn.relu(v66) # /* ty=Tensor[(1, 384, 40, 30), float32] */;
    v68 = relay.nn.conv2d(v67, params[140], padding=[1, 1, 1, 1], groups=384, kernel_size=[3, 3]) # /* ty=Tensor[(1, 384, 40, 30), float32] */;
    v69 = relay.nn.conv2d(v68, params[142], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 64, 40, 30), float32] */;
    v70 = relay.nn.batch_norm(v69, params[144], params[145], params[146], params[147]) # /* ty=(Tensor[(1, 64, 40, 30), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
    v71 = v70[0];
    v72 = relay.add(v71, v63) # /* ty=Tensor[(1, 64, 40, 30), float32] */;
    v73 = relay.nn.conv2d(v72, params[150], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 384, 40, 30), float32] */;
    v74 = relay.nn.batch_norm(v73, params[152], params[153], params[154], params[155]) # /* ty=(Tensor[(1, 384, 40, 30), float32], Tensor[(384), float32], Tensor[(384), float32]) */;
    v75 = v74[0];
    v76 = relay.nn.relu(v75) # /* ty=Tensor[(1, 384, 40, 30), float32] */;
    v77 = relay.nn.conv2d(v76, params[158], padding=[1, 1, 1, 1], groups=384, kernel_size=[3, 3]) # /* ty=Tensor[(1, 384, 40, 30), float32] */;
    v78 = relay.nn.conv2d(v77, params[160], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 64, 40, 30), float32] */;
    v79 = relay.nn.batch_norm(v78, params[162], params[163], params[164], params[165]) # /* ty=(Tensor[(1, 64, 40, 30), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
    v80 = v79[0];
    v81 = relay.add(v80, v72) # /* ty=Tensor[(1, 64, 40, 30), float32] */;
    v82 = relay.nn.conv2d(v81, params[168], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 384, 40, 30), float32] */;
    v83 = relay.nn.batch_norm(v82, params[170], params[171], params[172], params[173]) # /* ty=(Tensor[(1, 384, 40, 30), float32], Tensor[(384), float32], Tensor[(384), float32]) */;
    v84 = v83[0];
    v85 = relay.nn.relu(v84) # /* ty=Tensor[(1, 384, 40, 30), float32] */;
    v86 = relay.nn.conv2d(v85, params[176], padding=[1, 1, 1, 1], groups=384, kernel_size=[3, 3]) # /* ty=Tensor[(1, 384, 40, 30), float32] */;
    v87 = relay.nn.conv2d(v86, params[178], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 64, 40, 30), float32] */;
    v88 = relay.nn.batch_norm(v87, params[180], params[181], params[182], params[183]) # /* ty=(Tensor[(1, 64, 40, 30), float32], Tensor[(64), float32], Tensor[(64), float32]) */;
    v89 = v88[0];
    v90 = relay.add(v89, v81) # /* ty=Tensor[(1, 64, 40, 30), float32] */;
    v91 = relay.nn.conv2d(v90, params[186], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 384, 40, 30), float32] */;
    v92 = relay.nn.batch_norm(v91, params[188], params[189], params[190], params[191]) # /* ty=(Tensor[(1, 384, 40, 30), float32], Tensor[(384), float32], Tensor[(384), float32]) */;
    v93 = v92[0];
    v94 = relay.nn.relu(v93) # /* ty=Tensor[(1, 384, 40, 30), float32] */;
    v95 = relay.nn.conv2d(v94, params[194], padding=[1, 1, 1, 1], groups=384, kernel_size=[3, 3]) # /* ty=Tensor[(1, 384, 40, 30), float32] */;
    v96 = relay.nn.conv2d(v95, params[196], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 96, 40, 30), float32] */;
    v97 = relay.nn.batch_norm(v96, params[198], params[199], params[200], params[201]) # /* ty=(Tensor[(1, 96, 40, 30), float32], Tensor[(96), float32], Tensor[(96), float32]) */;
    v98 = v97[0];
    v99 = relay.nn.conv2d(v98, params[203], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 576, 40, 30), float32] */;
    v100 = relay.nn.batch_norm(v99, params[205], params[206], params[207], params[208]) # /* ty=(Tensor[(1, 576, 40, 30), float32], Tensor[(576), float32], Tensor[(576), float32]) */;
    v101 = v100[0];
    v102 = relay.nn.relu(v101) # /* ty=Tensor[(1, 576, 40, 30), float32] */;
    v103 = relay.nn.conv2d(v102, params[211], padding=[1, 1, 1, 1], groups=576, kernel_size=[3, 3]) # /* ty=Tensor[(1, 576, 40, 30), float32] */;
    v104 = relay.nn.conv2d(v103, params[213], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 96, 40, 30), float32] */;
    v105 = relay.nn.batch_norm(v104, params[215], params[216], params[217], params[218]) # /* ty=(Tensor[(1, 96, 40, 30), float32], Tensor[(96), float32], Tensor[(96), float32]) */;
    v106 = v105[0];
    v107 = relay.add(v106, v98) # /* ty=Tensor[(1, 96, 40, 30), float32] */;
    v108 = relay.nn.conv2d(v107, params[221], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 576, 40, 30), float32] */;
    v109 = relay.nn.batch_norm(v108, params[223], params[224], params[225], params[226]) # /* ty=(Tensor[(1, 576, 40, 30), float32], Tensor[(576), float32], Tensor[(576), float32]) */;
    v110 = v109[0];
    v111 = relay.nn.relu(v110) # /* ty=Tensor[(1, 576, 40, 30), float32] */;
    v112 = relay.nn.conv2d(v111, params[229], padding=[1, 1, 1, 1], groups=576, kernel_size=[3, 3]) # /* ty=Tensor[(1, 576, 40, 30), float32] */;
    v113 = relay.nn.conv2d(v112, params[231], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 96, 40, 30), float32] */;
    v114 = relay.nn.batch_norm(v113, params[233], params[234], params[235], params[236]) # /* ty=(Tensor[(1, 96, 40, 30), float32], Tensor[(96), float32], Tensor[(96), float32]) */;
    v115 = v114[0];
    v116 = relay.add(v115, v107) # /* ty=Tensor[(1, 96, 40, 30), float32] */;
    v117 = relay.nn.conv2d(v116, params[239], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 96, 40, 30), float32] */;
    v118 = relay.nn.batch_norm(v117, params[241], params[242], params[243], params[244]) # /* ty=(Tensor[(1, 96, 40, 30), float32], Tensor[(96), float32], Tensor[(96), float32]) */;
    v119 = v118[0];
    v120 = relay.nn.relu(v119) # /* ty=Tensor[(1, 96, 40, 30), float32] */;
    v121 = relay.nn.conv2d(v120, params[247], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 16, 40, 30), float32] */;
    v122 = relay.nn.batch_norm(v121, params[249], params[250], params[251], params[252]) # /* ty=(Tensor[(1, 16, 40, 30), float32], Tensor[(16), float32], Tensor[(16), float32]) */;
    v123 = v122[0];

    return v123

def graph_heads(features, params): # /* ty=Tensor[(1, 16, 40, 30), float16] */;
    v123 = relay.cast(features, 'float32') # /* ty=Tensor[(1, 16, 40, 30), float32] */;
    v124 = relay.nn.conv2d(v123, params[254], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 96, 40, 30), float32] */;
    v125 = relay.nn.batch_norm(v124, params[256], params[257], params[258], params[259]) # /* ty=(Tensor[(1, 96, 40, 30), float32], Tensor[(96), float32], Tensor[(96), float32]) */;
    v126 = v125[0];
    v127 = relay.nn.relu(v126) # /* ty=Tensor[(1, 96, 40, 30), float32] */;
    v128 = relay.nn.conv2d(v127, params[262], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 96, 40, 30), float32] */;
    v129 = relay.nn.batch_norm(v128, params[264], params[265], params[266], params[267]) # /* ty=(Tensor[(1, 96, 40, 30), float32], Tensor[(96), float32], Tensor[(96), float32]) */;
    v130 = v129[0];
    v131 = relay.nn.conv2d(v130, params[269], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 576, 40, 30), float32] */;
    v132 = relay.nn.batch_norm(v131, params[271], params[272], params[273], params[274]) # /* ty=(Tensor[(1, 576, 40, 30), float32], Tensor[(576), float32], Tensor[(576), float32]) */;
    v133 = v132[0];
    v134 = relay.nn.relu(v133) # /* ty=Tensor[(1, 576, 40, 30), float32] */;
    v135 = relay.nn.conv2d(v134, params[277], padding=[1, 1, 1, 1], groups=576, kernel_size=[3, 3]) # /* ty=Tensor[(1, 576, 40, 30), float32] */;
    v136 = relay.nn.conv2d(v135, params[279], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 96, 40, 30), float32] */;
    v137 = relay.nn.batch_norm(v136, params[281], params[282], params[283], params[284]) # /* ty=(Tensor[(1, 96, 40, 30), float32], Tensor[(96), float32], Tensor[(96), float32]) */;
    v138 = v137[0];
    v139 = relay.add(v138, v130) # /* ty=Tensor[(1, 96, 40, 30), float32] */;
    v140 = relay.nn.conv2d(v139, params[287], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 576, 40, 30), float32] */;
    v141 = relay.nn.batch_norm(v140, params[289], params[290], params[291], params[292]) # /* ty=(Tensor[(1, 576, 40, 30), float32], Tensor[(576), float32], Tensor[(576), float32]) */;
    v142 = v141[0];
    v143 = relay.nn.relu(v142) # /* ty=Tensor[(1, 576, 40, 30), float32] */;
    v144 = relay.nn.conv2d(v143, params[295], padding=[1, 1, 1, 1], groups=576, kernel_size=[3, 3]) # /* ty=Tensor[(1, 576, 40, 30), float32] */;
    v145 = relay.nn.conv2d(v144, params[297], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 96, 40, 30), float32] */;
    v146 = relay.nn.batch_norm(v145, params[299], params[300], params[301], params[302]) # /* ty=(Tensor[(1, 96, 40, 30), float32], Tensor[(96), float32], Tensor[(96), float32]) */;
    v147 = v146[0];
    v148 = relay.add(v147, v139) # /* ty=Tensor[(1, 96, 40, 30), float32] */;
    v149 = relay.nn.conv2d(v148, params[305], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 576, 40, 30), float32] */;
    v150 = relay.nn.batch_norm(v149, params[307], params[308], params[309], params[310]) # /* ty=(Tensor[(1, 576, 40, 30), float32], Tensor[(576), float32], Tensor[(576), float32]) */;
    v151 = v150[0];
    v152 = relay.nn.relu(v151) # /* ty=Tensor[(1, 576, 40, 30), float32] */;
    v153 = relay.nn.conv2d(v152, params[313], padding=[1, 1, 1, 1], groups=576, kernel_size=[3, 3]) # /* ty=Tensor[(1, 576, 40, 30), float32] */;
    v154 = relay.nn.conv2d(v153, params[315], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 96, 40, 30), float32] */;
    v155 = relay.nn.batch_norm(v154, params[317], params[318], params[319], params[320]) # /* ty=(Tensor[(1, 96, 40, 30), float32], Tensor[(96), float32], Tensor[(96), float32]) */;
    v156 = v155[0];
    v157 = relay.add(v156, v148) # /* ty=Tensor[(1, 96, 40, 30), float32] */;
    v158 = relay.nn.conv2d(v157, params[326], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 60, 40, 30), float32] */;
    v159 = relay.nn.bias_add(v158, params[327]) # /* ty=Tensor[(1, 60, 40, 30), float32] */;
    v160 = relay.transpose(v159, axes=[0, 2, 3, 1]) # /* ty=Tensor[(1, 40, 30, 60), float32] */;
    v161 = relay.reshape(v160, newshape=[1, -1, 4]) # /* ty=Tensor[(1, 18000, 4), float32] */;
    v162 = relay.nn.conv2d(v157, params[323], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(1, 15, 40, 30), float32] */;
    v163 = relay.nn.bias_add(v162, params[324]) # /* ty=Tensor[(1, 15, 40, 30), float32] */;
    v164 = relay.transpose(v163, axes=[0, 2, 3, 1]) # /* ty=Tensor[(1, 40, 30, 15), float32] */;
    v165 = relay.nn.batch_flatten(v164) # /* ty=Tensor[(1, 18000), float32] */;
    v166 = relay.sigmoid(v165) # /* ty=Tensor[(1, 18000), float32] */;
    v167 = relay.topk(v166, k=200, axis=1, dtype="int32") # /* ty=(Tensor[(1, 200), float32], Tensor[(1, 200), int32]) */;
    v168 = v167[1];
    v169 = relay.squeeze(v168, axis=[0]) # /* ty=Tensor[(200), int32] */;
    v171 = relay.take(v161, v169, axis=1) # /* ty=Tensor[(1, 200, 4), float32] */;
    v172 = relay.squeeze(v171, axis=[0]) # /* ty=Tensor[(200, 4), float32] */;
    v174 = relay.take(v172, params[344], axis=1) # /* ty=Tensor[(200, 1), float32] */;
    v176 = relay.take(params[341], v169, axis=1) # /* ty=Tensor[(1, 200, 4), float32] */;
    v177 = relay.reshape(v176, newshape=[200, 4]) # /* ty=Tensor[(200, 4), float32] */;
    v180 = relay.take(v177, params[359], axis=1) # /* ty=Tensor[(200, 1), float32] */;
    v182 = relay.take(v177, params[355], axis=1) # /* ty=Tensor[(200, 1), float32] */;
    v183 = relay.subtract(v180, v182) # /* ty=Tensor[(200, 1), float32] */;
    v184 = relay.add(v183, params[364]) # /* ty=Tensor[(200, 1), float32] */;
    v185 = relay.multiply(v174, v184) # /* ty=Tensor[(200, 1), float32] */;
    v186 = relay.multiply(params[369], v184) # /* ty=Tensor[(200, 1), float32] */;
    v187 = relay.add(v182, v186) # /* ty=Tensor[(200, 1), float32] */;
    v188 = relay.add(v185, v187) # /* ty=Tensor[(200, 1), float32] */;
    v190 = relay.take(v172, params[348], axis=1) # /* ty=Tensor[(200, 1), float32] */;
    v191 = relay.clip(v190, a_min=-inff, a_max=4.13517) # /* ty=Tensor[(200, 1), float32] */;
    v192 = relay.exp(v191) # /* ty=Tensor[(200, 1), float32] */;
    v193 = relay.multiply(v192, v184) # /* ty=Tensor[(200, 1), float32] */;
    v194 = relay.multiply(v193, params[385]) # /* ty=Tensor[(200, 1), float32] */;
    v195 = relay.subtract(v188, v194) # /* ty=Tensor[(200, 1), float32] */;
    v197 = relay.take(v172, params[346], axis=1) # /* ty=Tensor[(200, 1), float32] */;
    v199 = relay.take(v177, params[361], axis=1) # /* ty=Tensor[(200, 1), float32] */;
    v201 = relay.take(v177, params[357], axis=1) # /* ty=Tensor[(200, 1), float32] */;
    v202 = relay.subtract(v199, v201) # /* ty=Tensor[(200, 1), float32] */;
    v203 = relay.add(v202, params[367]) # /* ty=Tensor[(200, 1), float32] */;
    v204 = relay.multiply(v197, v203) # /* ty=Tensor[(200, 1), float32] */;
    v205 = relay.multiply(params[372], v203) # /* ty=Tensor[(200, 1), float32] */;
    v206 = relay.add(v201, v205) # /* ty=Tensor[(200, 1), float32] */;
    v207 = relay.add(v204, v206) # /* ty=Tensor[(200, 1), float32] */;
    v209 = relay.take(v172, params[350], axis=1) # /* ty=Tensor[(200, 1), float32] */;
    v210 = relay.clip(v209, a_min=-inff, a_max=4.13517) # /* ty=Tensor[(200, 1), float32] */;
    v211 = relay.exp(v210) # /* ty=Tensor[(200, 1), float32] */;
    v212 = relay.multiply(v211, v203) # /* ty=Tensor[(200, 1), float32] */;
    v213 = relay.multiply(v212, params[388]) # /* ty=Tensor[(200, 1), float32] */;
    v214 = relay.subtract(v207, v213) # /* ty=Tensor[(200, 1), float32] */;
    v215 = relay.multiply(v193, params[391]) # /* ty=Tensor[(200, 1), float32] */;
    v216 = relay.add(v188, v215) # /* ty=Tensor[(200, 1), float32] */;
    v217 = relay.subtract(v216, params[394]) # /* ty=Tensor[(200, 1), float32] */;
    v218 = relay.multiply(v212, params[396]) # /* ty=Tensor[(200, 1), float32] */;
    v219 = relay.add(v207, v218) # /* ty=Tensor[(200, 1), float32] */;
    v220 = relay.subtract(v219, params[399]) # /* ty=Tensor[(200, 1), float32] */;
    v221 = (v195, v214, v217, v220);
    v222 = relay.concatenate(v221, axis=1) # /* ty=Tensor[(200, 4), float32] */;
    v224 = relay.take(v222, params[403], axis=1) # /* ty=Tensor[(200, 1), float32] */;
    v226 = relay.take(v222, params[407], axis=1) # /* ty=Tensor[(200, 1), float32] */;
    v227 = (v224, v226);
    v228 = relay.concatenate(v227, axis=1) # /* ty=Tensor[(200, 2), float32] */;
    v229 = relay.clip(v228, a_min=0.0, a_max=479.0) # /* ty=Tensor[(200, 2), float32] */;
    v230 = relay.expand_dims(v229, axis=2) # /* ty=Tensor[(200, 2, 1), float32] */;
    v232 = relay.take(v222, params[405], axis=1) # /* ty=Tensor[(200, 1), float32] */;
    v234 = relay.take(v222, params[409], axis=1) # /* ty=Tensor[(200, 1), float32] */;
    v235 = (v232, v234);
    v236 = relay.concatenate(v235, axis=1) # /* ty=Tensor[(200, 2), float32] */;
    v237 = relay.clip(v236, a_min=0.0, a_max=479.0) # /* ty=Tensor[(200, 2), float32] */;
    v238 = relay.expand_dims(v237, axis=2) # /* ty=Tensor[(200, 2, 1), float32] */;
    v239 = (v230, v238);
    v240 = relay.concatenate(v239, axis=2) # /* ty=Tensor[(200, 2, 2), float32] */;
    v241 = relay.reshape(v240, newshape=[-1, 4]) # /* ty=Tensor[(200, 4), float32] */;
    v243 = relay.take(v241, params[420], axis=1) # /* ty=Tensor[(200, 1), float32] */;
    v245 = relay.take(v241, params[424], axis=1) # /* ty=Tensor[(200, 1), float32] */;
    v246 = relay.add(v243, v245) # /* ty=Tensor[(200, 1), float32] */;
    v247 = relay.divide(v246, params[429]) # /* ty=Tensor[(200, 1), float32] */;
    v249 = relay.take(v241, params[422], axis=1) # /* ty=Tensor[(200, 1), float32] */;
    v251 = relay.take(v241, params[426], axis=1) # /* ty=Tensor[(200, 1), float32] */;
    v252 = relay.add(v249, v251) # /* ty=Tensor[(200, 1), float32] */;
    v253 = relay.divide(v252, params[432]) # /* ty=Tensor[(200, 1), float32] */;
    v254 = relay.subtract(v245, v243) # /* ty=Tensor[(200, 1), float32] */;
    v255 = relay.add(v254, params[435]) # /* ty=Tensor[(200, 1), float32] */;
    v256 = relay.subtract(v251, v249) # /* ty=Tensor[(200, 1), float32] */;
    v257 = relay.add(v256, params[438]) # /* ty=Tensor[(200, 1), float32] */;
    v258 = (v247, v253, v255, v257);
    v259 = relay.concatenate(v258, axis=-1) # /* ty=Tensor[(200, 4), float32] */;
    v260 = v167[0];
    v261 = relay.squeeze(v260, axis=[0]) # /* ty=Tensor[(200), float32] */;
    v262 = relay.expand_dims(v261, axis=0) # /* ty=Tensor[(1, 200), float32] */;
    v263 = relay.expand_dims(v262, axis=0) # /* ty=Tensor[(1, 1, 200), float32] */;
    v264 = relay.transpose(v263, axes=[0, 2, 1]) # /* ty=Tensor[(1, 200, 1), float32] */;
    v265 = relay.argmax(v264, axis=[2], keepdims=True) # /* ty=Tensor[(1, 200, 1), int32] */;
    v267 = relay.cast(v265, dtype="float32") # /* ty=Tensor[(1, 200, 1), float32] */;
    v268 = relay.max(v264, axis=[2], keepdims=True) # /* ty=Tensor[(1, 200, 1), float32] */;
    v269 = relay.expand_dims(v259, axis=0) # /* ty=Tensor[(1, 200, 4), float32] */;
    v270 = relay.split(v269, indices_or_sections=4, axis=2) # /* ty=(Tensor[(1, 200, 1), float32], Tensor[(1, 200, 1), float32], Tensor[(1, 200, 1), float32], Tensor[(1, 200, 1), float32]) */;
    v271 = v270[0];
    v272 = v270[2];
    v273 = relay.divide(v272, relay.const(2.0)) # /* ty=Tensor[(1, 200, 1), float32] */;
    v274 = relay.subtract(v271, v273) # /* ty=Tensor[(1, 200, 1), float32] */;
    v275 = v270[1];
    v276 = v270[3];
    v277 = relay.divide(v276, relay.const(2.0)) # /* ty=Tensor[(1, 200, 1), float32] */;
    v278 = relay.subtract(v275, v277) # /* ty=Tensor[(1, 200, 1), float32] */;
    v279 = relay.divide(v272, relay.const(2.0)) # /* ty=Tensor[(1, 200, 1), float32] */;
    v280 = relay.add(v271, v279) # /* ty=Tensor[(1, 200, 1), float32] */;
    v281 = relay.divide(v276, relay.const(2.0)) # /* ty=Tensor[(1, 200, 1), float32] */;
    v282 = relay.add(v275, v281) # /* ty=Tensor[(1, 200, 1), float32] */;
    v283 = (v274, v278, v280, v282);
    v284 = relay.concatenate(v283, axis=2) # /* ty=Tensor[(1, 200, 4), float32] */;
    v285 = (v267, v268, v284);
    v286 = relay.concatenate(v285, axis=2) # /* ty=Tensor[(1, 200, 6), float32] */;
    v287 = relay.vision.non_max_suppression(v286, relay.const([200]), iou_threshold=0.5) # /* ty=Tensor[(1, 200), int32] */;
    v289 = relay.reshape(v287, newshape=[-1, 1]) # /* ty=Tensor[(200, 1), int32] */;
    v290 = relay.zeros_like(v289) # /* ty=Tensor[(200, 1), int32] */;
    v292 = relay.squeeze(v265, axis=[0]) # /* ty=Tensor[(200, 1), int32] */;
    v293 = relay.reshape(v287, newshape=[-1]) # /* ty=Tensor[(200), int32] */;
    v294 = relay.take(v292, v293, axis=0) # /* ty=Tensor[(200, 1), int32] */;
    v296 = relay.add(v294, relay.const(20)) # /* ty=Tensor[(200, 1), int32] */;
    v297 = (v290, v296, v289);
    v298 = relay.concatenate(v297, axis=1) # /* ty=Tensor[(200, 3), int32] */;
    v300 = relay.take(v298, params[448], axis=1) # /* ty=Tensor[(200, 1), int32] */;
    v301 = relay.squeeze(v300, axis=[1]) # /* ty=Tensor[(200), int32] */;
    v304 = relay.take(v259, v301, axis=0) # /* ty=Tensor[(200, 4), float32] */;
    v305 = relay.split(v304, indices_or_sections=[1, 2, 3], axis=-1) # /* ty=(Tensor[(200, 1), float32], Tensor[(200, 1), float32], Tensor[(200, 1), float32], Tensor[(200, 1), float32]) */;
    v306 = v305[0];
    v307 = v305[2];
    v308 = relay.divide(v307, params[457]) # /* ty=Tensor[(200, 1), float32] */;
    v309 = relay.subtract(v306, v308) # /* ty=Tensor[(200, 1), float32] */;
    v310 = v305[1];
    v311 = v305[3];
    v312 = relay.divide(v311, params[460]) # /* ty=Tensor[(200, 1), float32] */;
    v313 = relay.subtract(v310, v312) # /* ty=Tensor[(200, 1), float32] */;
    v314 = relay.divide(v307, params[463]) # /* ty=Tensor[(200, 1), float32] */;
    v315 = relay.add(v306, v314) # /* ty=Tensor[(200, 1), float32] */;
    v316 = relay.divide(v311, params[466]) # /* ty=Tensor[(200, 1), float32] */;
    v317 = relay.add(v310, v316) # /* ty=Tensor[(200, 1), float32] */;
    v318 = (v309, v313, v315, v317);
    v319 = relay.concatenate(v318, axis=-1) # /* ty=Tensor[(200, 4), float32] */;
    v320 = relay.strided_slice(v319, begin=[0], end=[8]) # /* ty=Tensor[(8, 4), float32] */;
    v321 = (params[475], v320);
    v322 = relay.concatenate(v321, axis=1) # /* ty=Tensor[(8, 5), float32] */;
    v324 = relay.take(v322, params[477], axis=1) # /* ty=Tensor[(8, 1), float32] */;
    v325 = relay.squeeze(v324, axis=[1]) # /* ty=Tensor[(8), float32] */;
    v327 = relay.expand_dims(v325, axis=1) # /* ty=Tensor[(8, 1), float32] */;
    v330 = relay.take(v322, params[481], axis=1) # /* ty=Tensor[(8, 4), float32] */;
    v331 = (v327, v330);
    v332 = relay.concatenate(v331, axis=1) # /* ty=Tensor[(8, 5), float32] */;
    v333 = relay.vision.roi_align(v130, v332, (6, 6), 0.0625, 2) # /* ty=Tensor[(8, 96, 6, 6), float32] */;
    v334 = relay.nn.conv2d(v333, params[484], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 384, 6, 6), float32] */;
    v335 = relay.nn.batch_norm(v334, params[486], params[487], params[488], params[489]) # /* ty=(Tensor[(8, 384, 6, 6), float32], Tensor[(384), float32], Tensor[(384), float32]) */;
    v336 = v335[0];
    v337 = relay.nn.relu(v336) # /* ty=Tensor[(8, 384, 6, 6), float32] */;
    v338 = relay.nn.conv2d(v337, params[492], strides=[2, 2], padding=[1, 1, 1, 1], groups=384, kernel_size=[3, 3]) # /* ty=Tensor[(8, 384, 3, 3), float32] */;
    v339 = relay.nn.conv2d(v338, params[494], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 160, 3, 3), float32] */;
    v340 = relay.nn.batch_norm(v339, params[496], params[497], params[498], params[499]) # /* ty=(Tensor[(8, 160, 3, 3), float32], Tensor[(160), float32], Tensor[(160), float32]) */;
    v341 = v340[0];
    v342 = relay.nn.conv2d(v341, params[501], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 960, 3, 3), float32] */;
    v343 = relay.nn.batch_norm(v342, params[503], params[504], params[505], params[506]) # /* ty=(Tensor[(8, 960, 3, 3), float32], Tensor[(960), float32], Tensor[(960), float32]) */;
    v344 = v343[0];
    v345 = relay.nn.relu(v344) # /* ty=Tensor[(8, 960, 3, 3), float32] */;
    v346 = relay.nn.conv2d(v345, params[509], padding=[1, 1, 1, 1], groups=960, kernel_size=[3, 3]) # /* ty=Tensor[(8, 960, 3, 3), float32] */;
    v347 = relay.nn.conv2d(v346, params[511], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 160, 3, 3), float32] */;
    v348 = relay.nn.batch_norm(v347, params[513], params[514], params[515], params[516]) # /* ty=(Tensor[(8, 160, 3, 3), float32], Tensor[(160), float32], Tensor[(160), float32]) */;
    v349 = v348[0];
    v350 = relay.add(v349, v341) # /* ty=Tensor[(8, 160, 3, 3), float32] */;
    v351 = relay.nn.conv2d(v350, params[519], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 960, 3, 3), float32] */;
    v352 = relay.nn.batch_norm(v351, params[521], params[522], params[523], params[524]) # /* ty=(Tensor[(8, 960, 3, 3), float32], Tensor[(960), float32], Tensor[(960), float32]) */;
    v353 = v352[0];
    v354 = relay.nn.relu(v353) # /* ty=Tensor[(8, 960, 3, 3), float32] */;
    v355 = relay.nn.conv2d(v354, params[527], padding=[1, 1, 1, 1], groups=960, kernel_size=[3, 3]) # /* ty=Tensor[(8, 960, 3, 3), float32] */;
    v356 = relay.nn.conv2d(v355, params[529], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 160, 3, 3), float32] */;
    v357 = relay.nn.batch_norm(v356, params[531], params[532], params[533], params[534]) # /* ty=(Tensor[(8, 160, 3, 3), float32], Tensor[(160), float32], Tensor[(160), float32]) */;
    v358 = v357[0];
    v359 = relay.add(v358, v350) # /* ty=Tensor[(8, 160, 3, 3), float32] */;
    v360 = relay.nn.conv2d(v359, params[537], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 960, 3, 3), float32] */;
    v361 = relay.nn.batch_norm(v360, params[539], params[540], params[541], params[542]) # /* ty=(Tensor[(8, 960, 3, 3), float32], Tensor[(960), float32], Tensor[(960), float32]) */;
    v362 = v361[0];
    v363 = relay.nn.relu(v362) # /* ty=Tensor[(8, 960, 3, 3), float32] */;
    v364 = relay.nn.conv2d(v363, params[545], padding=[1, 1, 1, 1], groups=960, kernel_size=[3, 3]) # /* ty=Tensor[(8, 960, 3, 3), float32] */;
    v365 = relay.nn.conv2d(v364, params[547], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 240, 3, 3), float32] */;
    v366 = relay.nn.batch_norm(v365, params[549], params[550], params[551], params[552]) # /* ty=(Tensor[(8, 240, 3, 3), float32], Tensor[(240), float32], Tensor[(240), float32]) */;
    v367 = v366[0];
    v368 = relay.nn.global_avg_pool2d(v367) # /* ty=Tensor[(8, 240, 1, 1), float32] */;
    v369 = relay.nn.batch_flatten(v368) # /* ty=Tensor[(8, 240), float32] */;
    v370 = relay.nn.batch_flatten(v369) # /* ty=Tensor[(8, 240), float32] */;
    v371 = relay.multiply(relay.const(1.0), v370) # /* ty=Tensor[(8, 240), float32] */;
    v372 = relay.transpose(params[559], axes=[1, 0]) # /* ty=Tensor[(324, 240), float32] */;
    v373 = relay.nn.dense(v371, v372, units=324) # /* ty=Tensor[(8, 324), float32] */;
    v374 = relay.multiply(relay.const(1.0), params[560]) # /* ty=Tensor[(324), float32] */;
    v375 = relay.nn.bias_add(v373, v374) # /* ty=Tensor[(8, 324), float32] */;
    v376 = relay.reshape(v375, newshape=[8, -1, 4]) # /* ty=Tensor[(8, 81, 4), float32] */;
    v378 = relay.take(v376, params[565], axis=2) # /* ty=Tensor[(8, 81, 1), float32] */;
    v379 = relay.squeeze(v378, axis=[2]) # /* ty=Tensor[(8, 81), float32] */;
    v380 = relay.divide(v379, params[598]) # /* ty=Tensor[(8, 81), float32] */;
    v383 = relay.take(v320, params[582], axis=1) # /* ty=Tensor[(8, 1), float32] */;
    v385 = relay.take(v320, params[578], axis=1) # /* ty=Tensor[(8, 1), float32] */;
    v386 = relay.subtract(v383, v385) # /* ty=Tensor[(8, 1), float32] */;
    v387 = relay.add(v386, params[587]) # /* ty=Tensor[(8, 1), float32] */;
    v388 = relay.multiply(v380, v387) # /* ty=Tensor[(8, 81), float32] */;
    v389 = relay.multiply(params[592], v387) # /* ty=Tensor[(8, 1), float32] */;
    v390 = relay.add(v385, v389) # /* ty=Tensor[(8, 1), float32] */;
    v391 = relay.add(v388, v390) # /* ty=Tensor[(8, 81), float32] */;
    v393 = relay.take(v376, params[571], axis=2) # /* ty=Tensor[(8, 81, 1), float32] */;
    v394 = relay.squeeze(v393, axis=[2]) # /* ty=Tensor[(8, 81), float32] */;
    v395 = relay.divide(v394, params[602]) # /* ty=Tensor[(8, 81), float32] */;
    v396 = relay.clip(v395, a_min=-inff, a_max=4.13517) # /* ty=Tensor[(8, 81), float32] */;
    v397 = relay.exp(v396) # /* ty=Tensor[(8, 81), float32] */;
    v398 = relay.multiply(v397, v387) # /* ty=Tensor[(8, 81), float32] */;
    v399 = relay.multiply(v398, params[616]) # /* ty=Tensor[(8, 81), float32] */;
    v400 = relay.subtract(v391, v399) # /* ty=Tensor[(8, 81), float32] */;
    v401 = relay.expand_dims(v400, axis=2) # /* ty=Tensor[(8, 81, 1), float32] */;
    v403 = relay.take(v376, params[568], axis=2) # /* ty=Tensor[(8, 81, 1), float32] */;
    v404 = relay.squeeze(v403, axis=[2]) # /* ty=Tensor[(8, 81), float32] */;
    v405 = relay.divide(v404, params[600]) # /* ty=Tensor[(8, 81), float32] */;
    v407 = relay.take(v320, params[584], axis=1) # /* ty=Tensor[(8, 1), float32] */;
    v409 = relay.take(v320, params[580], axis=1) # /* ty=Tensor[(8, 1), float32] */;
    v410 = relay.subtract(v407, v409) # /* ty=Tensor[(8, 1), float32] */;
    v411 = relay.add(v410, params[590]) # /* ty=Tensor[(8, 1), float32] */;
    v412 = relay.multiply(v405, v411) # /* ty=Tensor[(8, 81), float32] */;
    v413 = relay.multiply(params[595], v411) # /* ty=Tensor[(8, 1), float32] */;
    v414 = relay.add(v409, v413) # /* ty=Tensor[(8, 1), float32] */;
    v415 = relay.add(v412, v414) # /* ty=Tensor[(8, 81), float32] */;
    v417 = relay.take(v376, params[574], axis=2) # /* ty=Tensor[(8, 81, 1), float32] */;
    v418 = relay.squeeze(v417, axis=[2]) # /* ty=Tensor[(8, 81), float32] */;
    v419 = relay.divide(v418, params[604]) # /* ty=Tensor[(8, 81), float32] */;
    v420 = relay.clip(v419, a_min=-inff, a_max=4.13517) # /* ty=Tensor[(8, 81), float32] */;
    v421 = relay.exp(v420) # /* ty=Tensor[(8, 81), float32] */;
    v422 = relay.multiply(v421, v411) # /* ty=Tensor[(8, 81), float32] */;
    v423 = relay.multiply(v422, params[619]) # /* ty=Tensor[(8, 81), float32] */;
    v424 = relay.subtract(v415, v423) # /* ty=Tensor[(8, 81), float32] */;
    v425 = relay.expand_dims(v424, axis=2) # /* ty=Tensor[(8, 81, 1), float32] */;
    v426 = relay.multiply(v398, params[622]) # /* ty=Tensor[(8, 81), float32] */;
    v427 = relay.add(v391, v426) # /* ty=Tensor[(8, 81), float32] */;
    v428 = relay.subtract(v427, params[625]) # /* ty=Tensor[(8, 81), float32] */;
    v429 = relay.expand_dims(v428, axis=2) # /* ty=Tensor[(8, 81, 1), float32] */;
    v430 = relay.multiply(v422, params[627]) # /* ty=Tensor[(8, 81), float32] */;
    v431 = relay.add(v415, v430) # /* ty=Tensor[(8, 81), float32] */;
    v432 = relay.subtract(v431, params[630]) # /* ty=Tensor[(8, 81), float32] */;
    v433 = relay.expand_dims(v432, axis=2) # /* ty=Tensor[(8, 81, 1), float32] */;
    v434 = (v401, v425, v429, v433);
    v435 = relay.concatenate(v434, axis=2) # /* ty=Tensor[(8, 81, 4), float32] */;
    v436 = relay.reshape(v435, newshape=[-1, 4]) # /* ty=Tensor[(648, 4), float32] */;
    v438 = relay.take(v436, params[641], axis=1) # /* ty=Tensor[(648, 1), float32] */;
    v440 = relay.take(v436, params[645], axis=1) # /* ty=Tensor[(648, 1), float32] */;
    v441 = (v438, v440);
    v442 = relay.concatenate(v441, axis=1) # /* ty=Tensor[(648, 2), float32] */;
    v443 = relay.clip(v442, a_min=0.0, a_max=479.0) # /* ty=Tensor[(648, 2), float32] */;
    v444 = relay.expand_dims(v443, axis=2) # /* ty=Tensor[(648, 2, 1), float32] */;
    v446 = relay.take(v436, params[643], axis=1) # /* ty=Tensor[(648, 1), float32] */;
    v448 = relay.take(v436, params[647], axis=1) # /* ty=Tensor[(648, 1), float32] */;
    v449 = (v446, v448);
    v450 = relay.concatenate(v449, axis=1) # /* ty=Tensor[(648, 2), float32] */;
    v451 = relay.clip(v450, a_min=0.0, a_max=479.0) # /* ty=Tensor[(648, 2), float32] */;
    v452 = relay.expand_dims(v451, axis=2) # /* ty=Tensor[(648, 2, 1), float32] */;
    v453 = (v444, v452);
    v454 = relay.concatenate(v453, axis=2) # /* ty=Tensor[(648, 2, 2), float32] */;
    v455 = relay.reshape(v454, newshape=[-1, 4]) # /* ty=Tensor[(648, 4), float32] */;
    v457 = relay.take(v455, params[658], axis=1) # /* ty=Tensor[(648, 1), float32] */;
    v459 = relay.take(v455, params[662], axis=1) # /* ty=Tensor[(648, 1), float32] */;
    v460 = relay.add(v457, v459) # /* ty=Tensor[(648, 1), float32] */;
    v461 = relay.divide(v460, params[667]) # /* ty=Tensor[(648, 1), float32] */;
    v463 = relay.take(v455, params[660], axis=1) # /* ty=Tensor[(648, 1), float32] */;
    v465 = relay.take(v455, params[664], axis=1) # /* ty=Tensor[(648, 1), float32] */;
    v466 = relay.add(v463, v465) # /* ty=Tensor[(648, 1), float32] */;
    v467 = relay.divide(v466, params[670]) # /* ty=Tensor[(648, 1), float32] */;
    v468 = relay.subtract(v459, v457) # /* ty=Tensor[(648, 1), float32] */;
    v469 = relay.add(v468, params[673]) # /* ty=Tensor[(648, 1), float32] */;
    v470 = relay.subtract(v465, v463) # /* ty=Tensor[(648, 1), float32] */;
    v471 = relay.add(v470, params[676]) # /* ty=Tensor[(648, 1), float32] */;
    v472 = (v461, v467, v469, v471);
    v473 = relay.concatenate(v472, axis=-1) # /* ty=Tensor[(648, 4), float32] */;
    v474 = relay.reshape(v473, newshape=[-1, 81, 4]) # /* ty=Tensor[(8, 81, 4), float32] */;
    v475 = relay.strided_slice(v474, begin=[0, 1], end=[2147483647, 2147483647]) # /* ty=Tensor[(8, 80, 4), float32] */;
    v476 = relay.reshape(v475, newshape=[640, 4]) # /* ty=Tensor[(640, 4), float32] */;
    v477 = relay.nn.batch_flatten(v369) # /* ty=Tensor[(8, 240), float32] */;
    v478 = relay.multiply(relay.const(1.0), v477) # /* ty=Tensor[(8, 240), float32] */;
    v479 = relay.transpose(params[556], axes=[1, 0]) # /* ty=Tensor[(81, 240), float32] */;
    v480 = relay.nn.dense(v478, v479, units=81) # /* ty=Tensor[(8, 81), float32] */;
    v481 = relay.multiply(relay.const(1.0), params[557]) # /* ty=Tensor[(81), float32] */;
    v482 = relay.nn.bias_add(v480, v481) # /* ty=Tensor[(8, 81), float32] */;
    v483 = relay.nn.softmax(v482, axis=1) # /* ty=Tensor[(8, 81), float32] */;
    v484 = relay.reshape(v483, newshape=[-1]) # /* ty=Tensor[(648), float32] */;
    v485 = relay.reshape(v484, newshape=[-1, 81]) # /* ty=Tensor[(8, 81), float32] */;
    v486 = relay.strided_slice(v485, begin=[0, 1], end=[2147483647, 2147483647]) # /* ty=Tensor[(8, 80), float32] */;
    v487 = relay.reshape(v486, newshape=[640]) # /* ty=Tensor[(640), float32] */;
    v488 = relay.expand_dims(v487, axis=1) # /* ty=Tensor[(640, 1), float32] */;
    v489 = (params[699], v488);
    v490 = relay.concatenate(v489, axis=1) # /* ty=Tensor[(640, 81), float32] */;
    v491 = relay.reshape(v490, newshape=[8, 6480]) # /* ty=Tensor[(8, 6480), float32] */;
    v492 = relay.strided_slice(v491, begin=[0, 80], end=[2147483647, 2147483647]) # /* ty=Tensor[(8, 6400), float32] */;
    v493 = relay.reshape(v492, newshape=[640, 80]) # /* ty=Tensor[(640, 80), float32] */;
    v494 = relay.transpose(v493, axes=[1, 0]) # /* ty=Tensor[(80, 640), float32] */;
    v495 = relay.expand_dims(v494, axis=0) # /* ty=Tensor[(1, 80, 640), float32] */;
    v496 = relay.transpose(v495, axes=[0, 2, 1]) # /* ty=Tensor[(1, 640, 80), float32] */;
    v497 = relay.argmax(v496, axis=[2], keepdims=True) # /* ty=Tensor[(1, 640, 1), int32] */;
    v499 = relay.cast(v497, dtype="float32") # /* ty=Tensor[(1, 640, 1), float32] */;
    v500 = relay.max(v496, axis=[2], keepdims=True) # /* ty=Tensor[(1, 640, 1), float32] */;
    v501 = relay.expand_dims(v476, axis=0) # /* ty=Tensor[(1, 640, 4), float32] */;
    v502 = relay.split(v501, indices_or_sections=4, axis=2) # /* ty=(Tensor[(1, 640, 1), float32], Tensor[(1, 640, 1), float32], Tensor[(1, 640, 1), float32], Tensor[(1, 640, 1), float32]) */;
    v503 = v502[0];
    v504 = v502[2];
    v505 = relay.divide(v504, relay.const(2.0)) # /* ty=Tensor[(1, 640, 1), float32] */;
    v506 = relay.subtract(v503, v505) # /* ty=Tensor[(1, 640, 1), float32] */;
    v507 = v502[1];
    v508 = v502[3];
    v509 = relay.divide(v508, relay.const(2.0)) # /* ty=Tensor[(1, 640, 1), float32] */;
    v510 = relay.subtract(v507, v509) # /* ty=Tensor[(1, 640, 1), float32] */;
    v511 = relay.divide(v504, relay.const(2.0)) # /* ty=Tensor[(1, 640, 1), float32] */;
    v512 = relay.add(v503, v511) # /* ty=Tensor[(1, 640, 1), float32] */;
    v513 = relay.divide(v508, relay.const(2.0)) # /* ty=Tensor[(1, 640, 1), float32] */;
    v514 = relay.add(v507, v513) # /* ty=Tensor[(1, 640, 1), float32] */;
    v515 = (v506, v510, v512, v514);
    v516 = relay.concatenate(v515, axis=2) # /* ty=Tensor[(1, 640, 4), float32] */;
    v517 = (v499, v500, v516);
    v518 = relay.concatenate(v517, axis=2) # /* ty=Tensor[(1, 640, 6), float32] */;
    v519 = relay.vision.non_max_suppression(v518, relay.const([640]), iou_threshold=0.5) # /* ty=Tensor[(1, 640), int32] */;
    v521 = relay.reshape(v519, newshape=[-1, 1]) # /* ty=Tensor[(640, 1), int32] */;
    v522 = relay.zeros_like(v521) # /* ty=Tensor[(640, 1), int32] */;
    v524 = relay.squeeze(v497, axis=[0]) # /* ty=Tensor[(640, 1), int32] */;
    v525 = relay.reshape(v519, newshape=[-1]) # /* ty=Tensor[(640), int32] */;
    v526 = relay.take(v524, v525, axis=0) # /* ty=Tensor[(640, 1), int32] */;
    v528 = relay.add(v526, relay.const(20)) # /* ty=Tensor[(640, 1), int32] */;
    v529 = (v522, v528, v521);
    v530 = relay.concatenate(v529, axis=1) # /* ty=Tensor[(640, 3), int32] */;
    v532 = relay.take(v530, params[717], axis=1) # /* ty=Tensor[(640, 1), int32] */;
    v533 = relay.squeeze(v532, axis=[1]) # /* ty=Tensor[(640), int32] */;
    v536 = relay.take(v476, v533, axis=0) # /* ty=Tensor[(640, 4), float32] */;
    v539 = relay.take(v487, v533, axis=0) # /* ty=Tensor[(640), float32] */;
    v540 = relay.topk(v539, k=8, axis=0, dtype="int32") # /* ty=(Tensor[(8), float32], Tensor[(8), int32]) */;
    v541 = v540[1];
    v544 = relay.take(v536, v541, axis=0) # /* ty=Tensor[(8, 4), float32] */;
    v545 = relay.split(v544, indices_or_sections=[1, 2, 3], axis=-1) # /* ty=(Tensor[(8, 1), float32], Tensor[(8, 1), float32], Tensor[(8, 1), float32], Tensor[(8, 1), float32]) */;
    v546 = v545[0];
    v547 = v545[2];
    v548 = relay.divide(v547, params[740]) # /* ty=Tensor[(8, 1), float32] */;
    v549 = relay.subtract(v546, v548) # /* ty=Tensor[(8, 1), float32] */;
    v550 = v545[1];
    v551 = v545[3];
    v552 = relay.divide(v551, params[743]) # /* ty=Tensor[(8, 1), float32] */;
    v553 = relay.subtract(v550, v552) # /* ty=Tensor[(8, 1), float32] */;
    v554 = relay.divide(v547, params[746]) # /* ty=Tensor[(8, 1), float32] */;
    v555 = relay.add(v546, v554) # /* ty=Tensor[(8, 1), float32] */;
    v556 = relay.divide(v551, params[749]) # /* ty=Tensor[(8, 1), float32] */;
    v557 = relay.add(v550, v556) # /* ty=Tensor[(8, 1), float32] */;
    v558 = (v549, v553, v555, v557);
    boxes = relay.concatenate(v558, axis=-1) # /* ty=Tensor[(8, 4), float32] */;
    v562 = relay.take(params[697], v533, axis=0) # /* ty=Tensor[(640), int32] */;
    labels = relay.take(v562, v541, axis=0) # /* ty=Tensor[(8), int32] */;
    scores = relay.take(v539, v541, axis=0) # /* ty=Tensor[(8), float32] */;
    v569 = (params[753], boxes);
    v570 = relay.concatenate(v569, axis=1) # /* ty=Tensor[(8, 5), float32] */;
    v572 = relay.take(v570, params[755], axis=1) # /* ty=Tensor[(8, 1), float32] */;
    v573 = relay.squeeze(v572, axis=[1]) # /* ty=Tensor[(8), float32] */;
    v575 = relay.expand_dims(v573, axis=1) # /* ty=Tensor[(8, 1), float32] */;
    v578 = relay.take(v570, params[759], axis=1) # /* ty=Tensor[(8, 4), float32] */;
    v579 = (v575, v578);
    v580 = relay.concatenate(v579, axis=1) # /* ty=Tensor[(8, 5), float32] */;
    v581 = relay.vision.roi_align(v130, v580, (6, 6), 0.0625, 2) # /* ty=Tensor[(8, 96, 6, 6), float32] */;
    v582 = relay.nn.conv2d(v581, params[762], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 384, 6, 6), float32] */;
    v583 = relay.nn.batch_norm(v582, params[764], params[765], params[766], params[767]) # /* ty=(Tensor[(8, 384, 6, 6), float32], Tensor[(384), float32], Tensor[(384), float32]) */;
    v584 = v583[0];
    v585 = relay.nn.relu(v584) # /* ty=Tensor[(8, 384, 6, 6), float32] */;
    v586 = relay.nn.conv2d(v585, params[770], padding=[1, 1, 1, 1], groups=384, kernel_size=[3, 3]) # /* ty=Tensor[(8, 384, 6, 6), float32] */;
    v587 = relay.nn.conv2d(v586, params[772], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 160, 6, 6), float32] */;
    v588 = relay.nn.batch_norm(v587, params[774], params[775], params[776], params[777]) # /* ty=(Tensor[(8, 160, 6, 6), float32], Tensor[(160), float32], Tensor[(160), float32]) */;
    v589 = v588[0];
    v590 = relay.nn.conv2d(v589, params[779], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 960, 6, 6), float32] */;
    v591 = relay.nn.batch_norm(v590, params[781], params[782], params[783], params[784]) # /* ty=(Tensor[(8, 960, 6, 6), float32], Tensor[(960), float32], Tensor[(960), float32]) */;
    v592 = v591[0];
    v593 = relay.nn.relu(v592) # /* ty=Tensor[(8, 960, 6, 6), float32] */;
    v594 = relay.nn.conv2d(v593, params[787], padding=[1, 1, 1, 1], groups=960, kernel_size=[3, 3]) # /* ty=Tensor[(8, 960, 6, 6), float32] */;
    v595 = relay.nn.conv2d(v594, params[789], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 160, 6, 6), float32] */;
    v596 = relay.nn.batch_norm(v595, params[791], params[792], params[793], params[794]) # /* ty=(Tensor[(8, 160, 6, 6), float32], Tensor[(160), float32], Tensor[(160), float32]) */;
    v597 = v596[0];
    v598 = relay.add(v597, v589) # /* ty=Tensor[(8, 160, 6, 6), float32] */;
    v599 = relay.nn.conv2d(v598, params[797], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 960, 6, 6), float32] */;
    v600 = relay.nn.batch_norm(v599, params[799], params[800], params[801], params[802]) # /* ty=(Tensor[(8, 960, 6, 6), float32], Tensor[(960), float32], Tensor[(960), float32]) */;
    v601 = v600[0];
    v602 = relay.nn.relu(v601) # /* ty=Tensor[(8, 960, 6, 6), float32] */;
    v603 = relay.nn.conv2d(v602, params[805], padding=[1, 1, 1, 1], groups=960, kernel_size=[3, 3]) # /* ty=Tensor[(8, 960, 6, 6), float32] */;
    v604 = relay.nn.conv2d(v603, params[807], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 160, 6, 6), float32] */;
    v605 = relay.nn.batch_norm(v604, params[809], params[810], params[811], params[812]) # /* ty=(Tensor[(8, 160, 6, 6), float32], Tensor[(160), float32], Tensor[(160), float32]) */;
    v606 = v605[0];
    v607 = relay.add(v606, v598) # /* ty=Tensor[(8, 160, 6, 6), float32] */;
    v608 = relay.nn.conv2d(v607, params[815], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 960, 6, 6), float32] */;
    v609 = relay.nn.batch_norm(v608, params[817], params[818], params[819], params[820]) # /* ty=(Tensor[(8, 960, 6, 6), float32], Tensor[(960), float32], Tensor[(960), float32]) */;
    v610 = v609[0];
    v611 = relay.nn.relu(v610) # /* ty=Tensor[(8, 960, 6, 6), float32] */;
    v612 = relay.nn.conv2d(v611, params[823], padding=[1, 1, 1, 1], groups=960, kernel_size=[3, 3]) # /* ty=Tensor[(8, 960, 6, 6), float32] */;
    v613 = relay.nn.conv2d(v612, params[825], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 160, 6, 6), float32] */;
    v614 = relay.nn.batch_norm(v613, params[827], params[828], params[829], params[830]) # /* ty=(Tensor[(8, 160, 6, 6), float32], Tensor[(160), float32], Tensor[(160), float32]) */;
    v615 = v614[0];
    v616 = relay.add(v615, v607) # /* ty=Tensor[(8, 160, 6, 6), float32] */;
    v617 = relay.nn.conv2d(v616, params[833], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 480, 6, 6), float32] */;
    v618 = relay.nn.batch_norm(v617, params[835], params[836], params[837], params[838]) # /* ty=(Tensor[(8, 480, 6, 6), float32], Tensor[(480), float32], Tensor[(480), float32]) */;
    v619 = v618[0];
    v620 = relay.nn.relu(v619) # /* ty=Tensor[(8, 480, 6, 6), float32] */;
    v621 = relay.image.resize(v620, size=[12, 12], method="nearest_neighbor") # /* ty=Tensor[(8, 480, 12, 12), float32] */;
    v622 = relay.nn.conv2d(v621, params[843], padding=[1, 1, 1, 1], groups=480, kernel_size=[3, 3]) # /* ty=Tensor[(8, 480, 12, 12), float32] */;
    v623 = relay.nn.conv2d(v622, params[845], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 80, 12, 12), float32] */;
    v624 = relay.nn.batch_norm(v623, params[847], params[848], params[849], params[850]) # /* ty=(Tensor[(8, 80, 12, 12), float32], Tensor[(80), float32], Tensor[(80), float32]) */;
    v625 = v624[0];
    v626 = relay.nn.conv2d(v625, params[852], padding=[0, 0, 0, 0], kernel_size=[1, 1]) # /* ty=Tensor[(8, 81, 12, 12), float32] */;
    v627 = relay.nn.bias_add(v626, params[853]) # /* ty=Tensor[(8, 81, 12, 12), float32] */;
    v628 = relay.sigmoid(v627) # /* ty=Tensor[(8, 81, 12, 12), float32] */;
    v629 = relay.reshape(v628, newshape=[-1, 12, 12]) # /* ty=Tensor[(648, 12, 12), float32] */;
    v630 = relay.add(labels, params[858]) # /* ty=Tensor[(8), int32] */;
    v632 = relay.take(v629, v630, axis=0) # /* ty=Tensor[(8, 12, 12), float32] */;
    masks = relay.expand_dims(v632, axis=1) # /* ty=Tensor[(8, 1, 12, 12), float32] */;

    return boxes, labels, scores, masks


class BackboneGraphAnnotator(ExprMutator):
    def __init__(self, compiler):
        super(BackboneGraphAnnotator, self).__init__()
        self.compiler = compiler
        self.last_call = True

    def visit_call(self, call):
        curr_last = self.last_call
        self.last_call = False

        params = []
        for arg in call.args:
            param = super().visit(arg)
            if isinstance(param, (relay.expr.Var, relay.expr.Constant)):
                param = compiler_begin(param, self.compiler)
            params.append(param)

        new_call = relay.Call(call.op, params, call.attrs)
        if curr_last:
            new_call = compiler_end(new_call, self.compiler)
        return new_call


class HeadsGraphAnnotator(ExprMutator):
    def __init__(self, compiler):
        super(HeadsGraphAnnotator, self).__init__()
        self.compiler = compiler
        self.last_call = True

    def _mark_begin(self, expr):
        if isinstance(expr, (relay.expr.Var, relay.expr.Constant)):
            return compiler_begin(expr, self.compiler)
        else:
            return expr

    def visit_tuple(self, tup):
        curr_last = self.last_call
        self.last_call = False

        result = [self.visit(field) for field in tup.fields]

        if curr_last:
            result = [compiler_end(r, self.compiler) for r in result]

        return relay.Tuple(result)

    def visit_call(self, call):
        if call.op.name == "cast" and call.attrs.dtype == 'float16':
            return compiler_begin(call, self.compiler)
        if hasattr(call.attrs, 'compiler'):
            return compiler_begin(call, self.compiler)

        curr_last = self.last_call
        self.last_call = False

        params = []
        for arg in call.args:
            param = super().visit(arg)
            if isinstance(param, relay.expr.Tuple):
                param = relay.Tuple(tuple(self._mark_begin(p) for p in param))
            else:
                param = self._mark_begin(param)
            params.append(param)

        new_call = relay.Call(call.op, params, call.attrs)
        if curr_last:
            new_call = compiler_end(new_call, self.compiler)
        return new_call


def remove_batchnorm(expr):
    mod = tvm.IRModule()
    mod["main"] = relay.Function(relay.analysis.free_vars(expr), expr)

    seq = tvm.transform.Sequential([
        relay.transform.SimplifyInference(),
        relay.transform.FoldConstant(),
        relay.transform.FoldScaleAxis(),
    ])

    with relay.build_config(opt_level=3):
        mod = seq(mod)

    return mod["main"].body

def get_network(shape):
    image = relay.var("image", shape=shape, dtype='float32') # image: Tensor[(3, 640, 480), float32]
    param_path = "/".join([os.path.dirname(__file__), "mask_rcnn.params"])
    loaded_params = relay.load_param_dict(bytearray(open(param_path, "rb").read()))
    params = {}
    for k, v in loaded_params.items():
        params[int(k)] = relay.const(v)

    features = graph_backbone(image, params)
    features = remove_batchnorm(features)
    features = BackboneGraphAnnotator("coremlcompiler").visit(features)
    features = relay.cast(features, 'float16')

    boxes, labels, scores, masks = graph_heads(features, params)
    net = relay.Tuple((boxes, labels, scores, masks))
    net = HeadsGraphAnnotator("remote").visit(net)

    mod = tvm.IRModule()
    mod["main"] = relay.Function(relay.analysis.free_vars(net), net)

    return mod


if __name__ == "__main__":
    target = 'llvm'
    mod = get_network((3, 640, 480))
    mod = relay.transform.PartitionGraph()(mod)
    print(mod)

    with relay.build_config(opt_level=0):
        graph, lib, params = relay.build(mod, target)

    for m in lib.imported_modules:
        if m.type_key == "remote":
            remote = RemoteModule(m)
            symbol = remote.get_symbol()
            remote.connect("127.0.0.1", 9190)
            remote.build(mod, symbol, "llvm", True)
#            remote.build(mod, symbol, "llvm -target=x86_64-linux-gnu", True)
#            remote.build(mod, symbol, "cuda -libs=cublas,cudnn", False)

    ctx = tvm.cpu()
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input(**params)
    m.run()

    # evaluate
    ftimer = m.module.time_evaluator("run", ctx, number=1, repeat=10)
    prof_res = np.array(ftimer().results) * 1000
    print("%-19s (%s)" % ("%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))
