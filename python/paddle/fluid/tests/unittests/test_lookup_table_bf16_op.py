#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.tests.unittests.op_test import (
    OpTest, convert_float_to_uint16, check_out_dtype)
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle import enable_static, nn
from paddle.fluid import Program, program_guard


def _get_grad(weights, ids):
    w_shape = weights.shape
    w_grad = np.zeros((w_shape), dtype=np.float32)
    out_grad = weights[ids.flatten()]
    for i, idx in enumerate(ids):
        w_grad[idx] += out_grad[i]
    return w_grad


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestLookupTableBF16Op(OpTest):
    def setUp(self):
        self.op_type = "lookup_table"
        self.dtype = np.uint16

        table = np.random.random((17, 31)).astype("float32")
        self.ids = np.random.randint(0, 17, (4, 1)).astype("int64")

        self.mkldnn_data_type = "bfloat16"
        self.w_bf16 = convert_float_to_uint16(table)
        self.out = self.w_bf16[self.ids.flatten()]
        self.w_grad_bf16 = convert_float_to_uint16(_get_grad(table, self.ids))

        self.inputs = {'W': self.w_bf16, 'Ids': self.ids}
        self.outputs = {'Out': self.out}

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        self.check_grad(
            ['W'],
            'Out',
            no_grad_set=set('Ids'),
            check_dygraph=False,
            user_defined_grads=[self.w_grad_bf16],
            user_defined_grad_outputs=[self.out])


class TestEmbedOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            input_data = np.random.randint(0, 10, (4, 1)).astype("int64")

            def test_param_dtype():
                # dtype must be float32 or float64
                input2 = fluid.data(name='x2', shape=[4, 1], dtype='int64')
                fluid.layers.embedding(
                    input=input2, size=(10, 64), dtype='int64')

            self.assertRaises(TypeError, test_param_dtype)

            input3 = fluid.data(name='x3', shape=[4, 1], dtype='int64')
            fluid.layers.embedding(input=input3, size=(10, 64), dtype='uint16')


class TestOutDtype(unittest.TestCase):
    def test_dtype(self):
        api_fn = nn.functional.embedding
        check_out_dtype(
            api_fn,
            in_specs=[([10, 16], 'int64'), ([100, 64], )],
            expect_dtypes=['float32', 'float64'],
            target_index=1)


class TestLookupTableIsSparse(unittest.TestCase):
    def init_data(self):
        self.x_data = np.array([[1, 3, 0, 4, 7]]).astype("int64")
        self.y_data = np.array([[0.1, 0.3, 0, 0.4, 0.7]]).astype("float32")

    def get_w_grad(self, is_sparse):
        self.init_data()
        main_program = fluid.Program()
        with fluid.program_guard(main_program, fluid.Program()):
            x = fluid.layers.data(name='x', shape=[5, 1], dtype='int64')
            y_ = fluid.layers.data(name='y_', shape=[5, 1], dtype='float32')
            emb = fluid.layers.embedding(
                input=x,
                size=[10, 16],
                param_attr=fluid.ParamAttr(
                    name="emb_weight",
                    learning_rate=10,
                    initializer=fluid.initializer.NumpyArrayInitializer(
                        self.w_data)),
                is_sparse=is_sparse,
                dtype="uint16")
            y = fluid.layers.reduce_sum(emb, dim=-1)

            loss = fluid.layers.square_error_cost(input=y, label=y_)
            loss = fluid.layers.mean(loss)

            sgd_optimizer = fluid.optimizer.SGD(learning_rate=1e-4)
            sgd_optimizer.minimize(loss)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            ret = exe.run(feed={'x': self.x_data,
                                'y_': self.y_data},
                          fetch_list=['emb_weight'],
                          return_numpy=False)
            return np.array(ret[0])

    def test_w_grad(self):
        self.w_data = np.random.random(size=(10, 16)).astype("float32")
        w_grad = self.get_w_grad(False)
        w_grad_with_sparse = self.get_w_grad(True)
        self.check_grad(w_grad, w_grad_with_sparse)

    def check_grad(self, w_grad1, w_grad2, tolerance=1e-6):
        np.testing.assert_allclose(
            w_grad1, w_grad2, rtol=tolerance, atol=tolerance)


if __name__ == "__main__":
    enable_static()
    unittest.main()
