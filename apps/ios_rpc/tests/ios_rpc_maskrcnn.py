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

from tvm import rpc
from tvm import relay
from tvm.contrib import graph_runtime, util, xcode
from tvm.contrib.target.remote import RemoteModule
import mask_rcnn

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import os
import re
import sys


def preprocess(image, w, h):
    # Resize
    image = image.resize((w, h), Image.BILINEAR)

    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([103.53, 116.28, 123.675])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    std_vec = np.array([57.5, 57.5, 57.5])
    for i in range(image.shape[0]):
        image[i, :, :] /= std_vec[i]

    # Pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image

def display_objdetect_image(image, h, boxes, labels, scores, masks, score_threshold=0.1):
    # Resize boxes
    ratio = h / min(image.size[0], image.size[1])
    boxes /= ratio
    _, ax = plt.subplots(1, figsize=(12,9))
    image = np.array(image)

    for mask, box, label, score in zip(masks, boxes, labels, scores):
        if score <= score_threshold:
            break
        mask = mask[0, :, :, None]
        mask = cv2.resize(mask, (int(box[2])-int(box[0])+1, int(box[3])-int(box[1])+1))
        mask = mask > 0.5
        im_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        x_0 = max(int(box[0]), 0)
        x_1 = min(int(box[2]) + 1, image.shape[1])
        y_0 = max(int(box[1]), 0)
        y_1 = min(int(box[3]) + 1, image.shape[0])

        try:
            im_mask[int(y_0):int(y_1), int(x_0):int(x_1)] = mask[
                (y_0 - int(box[1])) : (y_1 - int(box[1])), (x_0 - int(box[0])) : (x_1 - int(box[0]))
            ]
        except Exception as e:
            print(e)

        im_mask = im_mask[:, :, None]
        contours, hierarchy = cv2.findContours(
            im_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        image = cv2.drawContours(image, contours, -1, 25, 3)

    ax.imshow(image)

    # Showing boxes with score > 0.7
    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b', facecolor='none')
            label = classes[label] + ':' + str(np.round(score, 2))
            ax.annotate(label, (box[0], box[1]), color='w', fontsize=12)
            print(label, box)
            ax.add_patch(rect)
    plt.savefig("figure.png")

classes_path = "/".join([os.path.dirname(__file__), "coco_classes.txt"])
classes = [line.rstrip('\n') for line in open(classes_path)]


######################################################################
# Execute on TVM
# ---------------------------------------------
# load the module back.

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
#arch = "arm64"
#sdk = "iphoneos"
arch = "x86_64"
sdk = "iphonesimulator"
target = "llvm -target=%s-apple-darwin" % arch

img_path = "/".join([os.path.dirname(__file__), "demo.jpg"])
img = Image.open(img_path)
HEIGHT = 640
WIDTH = 480
img = img.resize((WIDTH, HEIGHT), Image.BILINEAR)
img_data = preprocess(img, WIDTH, HEIGHT)

param_path = "/".join([os.path.dirname(__file__), "mask_rcnn.params"])
mask_rcnn_params = relay.load_param_dict(bytearray(open(param_path, "rb").read()))
func = mask_rcnn.get_network((3, HEIGHT, WIDTH))
func = relay.transform.PartitionGraph()(func)

def test_rpc_module():
    temp = util.tempdir()

    """
    Compile
    """
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(func, target, params=mask_rcnn_params)

    print(graph)
    for m in lib.imported_modules:
        if m.type_key == "remote":
            remote = RemoteModule(m)
            symbol = remote.get_symbol()
            remote.connect(proxy_host, 9190)
            remote.build(func, symbol, "llvm", True)
#            remote.build(func, symbol, "llvm -target=x86_64-linux-gnu", True)
#            remote.build(func, symbol, "cuda", False)

    path_dso = temp.relpath("deploy.dylib")
    lib.export_library(path_dso, xcode.create_dylib,
                     arch=arch, sdk=sdk)

    xcode.codesign(path_dso)

    # Start RPC test server that contains the compiled library.
    server = xcode.popen_test_rpc(proxy_host, proxy_port, key,
                                  destination=destination,
                                  libs=[path_dso])

    # connect to the proxy
    remote = rpc.connect(proxy_host, proxy_port, key=key)
    ctx = remote.cpu(0)
    lib = remote.load_module("deploy.dylib")
    m = graph_runtime.create(graph, lib, ctx)

    # set inputs
    m.set_input(**params)
    # execute
    m.run(image=img_data)
    boxes = m.get_output(0).asnumpy()
    labels = m.get_output(1).asnumpy()
    scores = m.get_output(2).asnumpy()
    masks = m.get_output(3).asnumpy()
    print(boxes.shape)
    print(labels.shape)
    print(scores.shape)
    print(masks.shape)
    display_objdetect_image(img, min(HEIGHT,WIDTH), boxes, labels, scores, masks, )

    # evaluate
    ftimer = m.module.time_evaluator("run", ctx, number=1, repeat=10)
    prof_res = np.array(ftimer().results) * 1000
    print("%-19s (%s)" % ("%.2f ms" % np.mean(prof_res), "%.2f ms" % np.std(prof_res)))

test_rpc_module()
