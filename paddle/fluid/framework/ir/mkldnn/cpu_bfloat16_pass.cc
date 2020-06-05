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
#include <vector>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/mkldnn/cpu_bfloat16_pass.h"

namespace paddle {
namespace framework {
namespace ir {

void UnlinkNodes(ir::Node* a, ir::Node* b) {
  a->outputs.erase(std::remove(a->outputs.begin(), a->outputs.end(), b),
                   a->outputs.end());
  b->inputs.erase(std::remove(b->inputs.begin(), b->inputs.end(), a),
                  b->inputs.end());
}

void CPUBFloat16Pass::ApplyImpl(ir::Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::BFloat16Ops bfloat16_ops{gpd.mutable_pattern(),
                                                 "bfloat16_ops"};
  bfloat16_ops();

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                            Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(op, op, bfloat16_ops);
    GET_IR_NODE_FROM_SUBGRAPH(op_out, op_out, bfloat16_ops);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, bfloat16_ops);

    if (op->Op()->GetAttrIfExists<bool>("use_bfloat16")) {
      if ((op->Op()->HasAttr("force_fp32_output") ||
           op->Op()->HasProtoAttr("force_fp32_output")) &&
          !op->Op()->GetAttrIfExists<bool>("fuse_residual_connection")) {
        op->Op()->SetAttr("force_fp32_output", true);
      } else if (op->Op()->Type() != "prior_box") {
        // Create dequantize input variable
        VarDesc dequantize_in_desc(patterns::PDNodeName("dequantize", "in"));
        auto* dequantize_in_node = g->CreateVarNode(&dequantize_in_desc);

        // create a dequantize op node for output.
        OpDesc deq_desc;
        deq_desc.SetType("dequantize");
        deq_desc.SetInput(
            "Input", std::vector<std::string>({dequantize_in_node->Name()}));
        deq_desc.SetOutput("Output",
                           std::vector<std::string>({op_out->Name()}));
        deq_desc.SetAttr("Scale", 1.0f);
        auto dequantize_op = g->CreateOpNode(&deq_desc);

        std::string op_output_name;
        for (auto name : op->Op()->OutputNames()) {
          std::cout << name << " ";
          for (auto output_name : op->Op()->Output(name)) {
            std::cout << output_name << " " << op_out->Name() << " ";
            if (output_name == op_out->Name()) op_output_name = name;
          }
        }

        PADDLE_ENFORCE_NE(
            op_output_name.empty(), true,
            platform::errors::NotFound(
                "Operator after operator should have input as op output"));

        op->Op()->SetOutput(op_output_name, std::vector<std::string>(
                                                {dequantize_in_node->Name()}));

        UnlinkNodes(op, op_out);
        IR_NODE_LINK_TO(op, dequantize_in_node);
        IR_NODE_LINK_TO(dequantize_in_node, dequantize_op);
        IR_NODE_LINK_TO(dequantize_op, op_out);
      }
    }
  };
  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_bfloat16_pass, paddle::framework::ir::CPUBFloat16Pass);
