#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np

import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle.fluid.tests.unittests.test_conv2d_op import conv2d_forward_naive, TestConv2dOp


def conv2d_forward_refer(input, filter, group, conv_param):
    out, in_n, out_h, out_w, out_c = conv2d_forward_naive(input, filter, group,
                                                          conv_param)
    return out


def copy_bits_from_float_to_uint16(f):
    return struct.unpack('<I', struct.pack('<f', f))[0] >> 16


def convert_float_to_uint16(float_list):
    new_output = []
    for first in float_list:
        for second in first:
            for third in second:
                for fourth in third:
                    new_output.append(
                        np.uint16(copy_bits_from_float_to_uint16(fourth)))

    return np.reshape(new_output, float_list.shape).view(np.uint16)


class TestConv2dBf16Op(TestConv2dOp):
    def setUp(self):
        self.op_type = "conv2d"
        self.use_cudnn = False
        self.exhaustive_search = False
        self.use_cuda = False
        self.use_mkldnn = True
        self.weight_type = np.float32
        self.input_type = np.float32
        self.use_mkldnn = True
        self.mkldnn_data_type = False
        self.force_fp32_output = False
        self.init_group()
        self.init_dilation()
        self.init_test_case()
        self.init_fuse_relu()
        self.init_fuse_residual()
        self.init_data_type()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        self.input = np.random.random(self.input_size).astype(self.input_type)
        self.filter = np.random.random(self.filter_size).astype(self.weight_type)
        conv_out, _, _, _, _ = conv2d_forward_naive(
            self.input, self.filter, self.groups, conv2d_param)
        self.conv_output_float = conv_out

        if self.force_fp32_output:
          self.outputs = {'Output': self.conv_output_float}
        else
          self.conv_output = convert_float_to_uint16(conv_out)
          self.outputs = {'Output': self.conv_output}

        self.inputs = {
            'Input':
            OpTest.np_dtype_to_fluid_dtype(input.astype(self.input_type)),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
        }
        if self.fuse_residual:
            self.inputs['ResidualData'] = OpTest.np_dtype_to_fluid_dtype(
                input_residual)

        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'dilations': self.dilations,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn,
            'data_format': self.data_format,
            'exhaustive_search': self.exhaustive_search,
            'fuse_activation': self.fuse_activation,
            'fuse_residual_connection': self.fuse_residual
        }

    def test_check_output(self):
        self.check_output_with_place(
            core.CPUPlace(), atol=0, check_dygraph=False)

    def test_check_grad(self):
        pass

    def test_check_grad_no_filter(self):
        pass

    def test_check_grad_no_input(self):
        pass

    def init_test_case(self):
        TestConv2dOp.init_test_case(self)
        self.input_size = [1, 1, 5, 5]  # NCHW
        f_c = self.input_size[1] // self.groups
        self.input_residual_size = [1, 2, 3, 3]
        self.filter_size = [2, f_c, 3, 3]

    def init_data_type(self):
        self.weight_type = np.float32
        self.input_type = np.float32

    def init_fuse_relu(self):
        self.fuse_activation = "relu"

    def init_fuse_residual(self):
        self.fuse_residual = True


#--------------------test conv2d u8 in and u8 out with residual fuse--------------------


class TestConv2d(TestConv2dBf16Op):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.input_residual_size = [2, 6, 3, 3]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]


class TestWithPad(TestConv2d):
    def init_test_case(self):
        TestConv2d.init_test_case(self)
        self.pad = [1, 1]
        self.input_residual_size = [2, 6, 5, 5]


class TestWithGroup(TestConv2d):
    def init_group(self):
        self.groups = 3


class TestWithStride(TestConv2dBf16Op):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]
        self.input_residual_size = [2, 6, 3, 3]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]


class TestWith1x1(TestConv2dBf16Op):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [1, 3, 5, 5]
        self.input_residual_size = [1, 6, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]


class TestWithInput1x1Filter1x1(TestConv2dBf16Op):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 1, 1]
        self.input_residual_size = [2, 6, 1, 1]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]

    def init_group(self):
        self.groups = 3


if __name__ == '__main__':
    unittest.main()
