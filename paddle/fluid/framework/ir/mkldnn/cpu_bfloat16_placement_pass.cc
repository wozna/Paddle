/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <unordered_set>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/mkldnn/cpu_bfloat16_placement_pass.h"

namespace paddle {
namespace framework {
namespace ir {

void CPUBFloat16PlacementPass::ApplyImpl(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::BFloat16Placement bfloat16_placement_pattern{gpd.mutable_pattern(),
                                                         "bfloat16_placement"};
  bfloat16_placement_pattern();

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(prev_op, prev_op, bfloat16_placement_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(prev_out, prev_out, bfloat16_placement_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(op, op, bfloat16_placement_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(op_out, op_out, bfloat16_placement_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, bfloat16_placement_pattern);
     //check if previous operator can be change to bfloat16
    if ((prev_op->Op()->HasAttr("use_bfloat16") || prev_op->Op()->HasProtoAttr("use_bfloat16") || 
         next_op->Op()->HasAttr("use_bfloat16") || next_op->Op()->HasProtoAttr("use_bfloat16")) 
    && (op->Op()->GetAttrIfExists<bool>("use_mkldnn") || op->Op()->Type() == "reshape2")) {
      op->Op()->SetAttr("use_bfloat16", true);
    } else if (op->Op()->Type() == "conv2d" &&
               (next_op->Op()->HasAttr("use_bfloat16") ||
                next_op->Op()->HasProtoAttr("use_bfloat16"))) {
      // check if it is convolution, because it can change it to the to
      // bfloat16, but we remove situation when it is only one convolution
      // between ops that cannot be change to bfloat16
      op->Op()->SetAttr("use_bfloat16", true);
    }
  };
  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_bfloat16_placement_pass,
              paddle::framework::ir::CPUBFloat16PlacementPass);
