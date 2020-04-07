/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/layer_norm_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

template <typename T>
class LayerNormMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* y = ctx.Output<Tensor>("Y");

    const float epsilon = ctx.Attr<float>("epsilon");
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
    const bool is_test = ctx.Attr<bool>("is_test");

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();

    auto src_tz = paddle::framework::vectorize(x->dims());
    PADDLE_ENFORCE_EQ(begin_norm_axis, (src_tz.size() - 1),
                      platform::errors::InvalidArgument(
                          "MKL-DNN Layer Norm supports only last logical "
                          "axis:%d as begin_norm_axis.",
                          (src_tz.size() - 1)));

    y->mutable_data<T>(ctx.GetPlace());
    const bool with_scaleshift = (scale && bias);
    dnnl::normalization_flags flags{};

    if (with_scaleshift) {
      flags |= dnnl::normalization_flags::use_scale_shift;
    }

    platform::LayerNormMKLDNNHandler<T> handler(
        src_tz, epsilon, flags, is_test, x->format(), dev_ctx, ctx.GetPlace(),
        ctx.OutputName("Y"));

    auto src_memory = handler.AcquireSrcMemory(x);
    auto dst_memory = handler.AcquireDstMemory(y);

    auto layer_norm_p = handler.AcquireForwardPrimitive();

    dnnl::stream astream(dev_ctx.GetEngine());
    std::unordered_map<int, dnnl::memory> args;

    args.insert({DNNL_ARG_SRC, *src_memory});
    args.insert({DNNL_ARG_DST, *dst_memory});

    if (!is_test) {
      auto* mean = ctx.Output<Tensor>("Mean");
      auto* var = ctx.Output<Tensor>("Variance");
      mean->mutable_data<T>(ctx.GetPlace());
      var->mutable_data<T>(ctx.GetPlace());

      auto mean_memory = handler.AcquireMeanMemory(mean);
      auto variance_memory = handler.AcquireVarianceMemory(var);

      args.insert({DNNL_ARG_MEAN, *mean_memory});
      args.insert({DNNL_ARG_VARIANCE, *variance_memory});
    }

    if (with_scaleshift) {
      auto scaleshift_memory = handler.AcquireScaleShiftMemory();
      if (scaleshift_memory == nullptr || !is_test) {
        std::vector<T> scaleshift_data;
        auto scale_tz = paddle::framework::vectorize(scale->dims());
        const unsigned int C = scale_tz[0];

        // MKLDNN requires a single piece of memory for scale and shift/bias data
        scaleshift_data.reserve(2 * C);
        scaleshift_data.insert(scaleshift_data.begin(), scale->data<float>(),
                              scale->data<float>() + C);

        scaleshift_data.insert(scaleshift_data.end(), bias->data<float>(),
                              bias->data<float>() + C);

        scaleshift_memory =
          handler.AcquireScaleShiftMemory(scaleshift_data.data());

      }
      args.insert({DNNL_ARG_SCALE_SHIFT, *scaleshift_memory});
    }

    layer_norm_p->execute(astream, args);
    astream.wait();

    y->set_layout(DataLayout::kMKLDNN);
    y->set_format(platform::GetMKLDNNFormat(*dst_memory));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(layer_norm, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::LayerNormMKLDNNOpKernel<float>,
                   ops::LayerNormMKLDNNOpKernel<paddle::platform::bfloat16>);
