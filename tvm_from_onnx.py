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
"""
Compile ONNX Models
===================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_

This article is an introductory tutorial to deploy ONNX models with Relay.

For us to begin with, ONNX package must be installed.

A quick solution is to install protobuf compiler, and

.. code-block:: bash

    pip install onnx --user

or please refer to offical site.
https://github.com/onnx/onnx
"""
import onnx
import numpy as np
import tvm
import tvm.relay as relay
from tvm.contrib.download import download_testdata

######################################################################
# Load pretrained ONNX model
# ---------------------------------------------
vgg_model = onnx.load("vgg11.onnx")
matrix_model = onnx.load("matrix11.onnx")
decoder_model = onnx.load("decoder11.onnx")

#print('The model is:\n{}'.format(matrix_model))
#onnx.checker.check_model(matrix_model)

######################################################################
# Load a test image
# ---------------------------------------------
from PIL import Image
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_path = download_testdata(img_url, 'cat.png', module='data')
img = Image.open(img_path).resize((224, 334)) 
img_ycbcr = img #.convert("YCbCr")  # convert to YCbCr
img_y, img_cb, img_cr = img_ycbcr.split()
x = np.array(img)

print(img.size)
######################################################################
# Compile the model with relay
# ---------------------------------------------
target = 'cuda'

original_name = '0'#'1'
style_name = '1'#'1'
shape_dict_vgg = {'input.1': (1, 3, 224, 334)}
mod_vgg, params_vgg = relay.frontend.from_onnx(vgg_model, shape_dict_vgg)

#tvm::runtime::Module::LoadFromFile("/sdcard/deploy_flag_lib.so").
#mod_vgg['main'] = relay.quantize.quantize(mod_vgg['main'], params_vgg)

with relay.build_config(opt_level=3):
    vgg_executor = relay.build_module.create_executor('graph', mod_vgg, tvm.gpu(0), target)

dtype = 'float32'
vgg_output = vgg_executor.evaluate()(tvm.nd.array(x.astype(dtype)), **params_vgg).asnumpy()

print(vgg_output.shape)
shape_dict_matrix = { 'input.1': (1,256,64,64),'1': (1,256,56,83) }# , '1': (1, 3, 224, 224)
mod_matrix, params_matrix = relay.frontend.from_onnx(matrix_model, shape_dict_matrix)

with relay.build_config(opt_level=0):
    matrix_executor = relay.build_module.create_executor('graph', mod_matrix, tvm.gpu(0), target)

dtype = 'float32'
matrix_output = matrix_executor.evaluate()(tvm.nd.array([vgg_output.astype(dtype), vgg_output.astype(dtype)]), **params_matrix).asnumpy()

shape_dict_decoder = {'input.1': matrix_output.shape}# , '1': (1, 3, 224, 224)
mod_decoder, params_decoder = relay.frontend.from_onnx(decoder_model, shape_dict_decoder)

with relay.build_config(opt_level=3): #maybe use 3 instead of 1 like the example for more optimalizations? will t work?
    decoder_executor = relay.build_module.create_executor('graph', mod_decoder, tvm.gpu(0), target)

dtype = 'float32'
decoder_output = decoder_executor.evaluate()(tvm.nd.array(matrix_output.astype(dtype)), **params_decoder).asnumpy()
#intrp.export_library("xyz.so")

#intrp.export_library("deploy_lib.so")

#with open("deploy_graph.json", "w") as fo:
#    fo.write(graph.json())
#with open("deploy_param.params", "wb") as fo:
#    fo.write(relay.compiler.save_param_dict(params))


from matplotlib import pyplot as plt
out_y = Image.fromarray(np.uint8((tvm_output[0, 0]).clip(0, 255)))
