# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
from inference_pass_test import InferencePassTest
import paddle.fluid as fluid
from paddle.fluid.tests.unittests.test_fusion_gru_op import fusion_gru
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import PassVersionChecker


class TestMKLDNNCpuQunatizePass(InferencePassTest):
    def setUp(self):
        self.init_data()
        with fluid.program_guard(self.main_program, self.startup_program):
            dict_dim, emb_dim = 128, 64
            data = fluid.data(
                name='step_data', shape=[None], dtype='int64', lod_level=1)
            emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
            hidden_dim = 512
            x = fluid.layers.fc(input=emb, size=hidden_dim * 3, bias_attr=False)
            hidden = fluid.layers.dynamic_gru(
                input=x,
                size=hidden_dim,
                bias_attr=True,
                origin_mode=False,
                is_reverse=True)

        batch = 16
        lod_tensor = fluid.LoDTensor()
        lod_tensor.set(np.random.randint(
            0, dict_dim, size=[batch]).astype("int64"),
                       fluid.CPUPlace())
        lod_tensor.set_lod([[0, batch]])
        self.feeds = {"step_data": lod_tensor}
        infer_tensor = fluid.core.PaddleTensor()
        infer_tensor.lod = lod_tensor.lod()
        infer_tensor.data = fluid.core.PaddleBuf(np.array(lod_tensor))
        infer_tensor.shape = lod_tensor.shape()
        infer_tensor.dtype = fluid.core.PaddleDType.INT64

        self.warmup_data = [infer_tensor]
        self.fetch_list = [hidden]

    def init_data(self):
        self.enable_mkldnn = True
        self.enable_mkldnn_quantizer = True

    def test_check_output(self):
        use_gpu = False
        self.check_output_with_option(use_gpu, flatten=True)
        self.assertTrue(PassVersionChecker.IsCompatible('cpu_quantize_pass'))
        self.assertTrue(
            PassVersionChecker.IsCompatible('cpu_quantize_squash_pass'))


if __name__ == "__main__":
    unittest.main()
