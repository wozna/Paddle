/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

// setting iterations to 0 means processing the whole dataset
namespace paddle {
namespace inference {
namespace analysis {

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model);
  cfg->DisableGpu();
  cfg->SwitchIrOptim(true);
  cfg->SwitchSpecifyInputNames(false);
  cfg->SetCpuMathLibraryNumThreads(FLAGS_cpu_num_threads);
  cfg->EnableMKLDNN();
  cfg->SwitchIrDebug();
}

std::vector<size_t> ReadObjectsNum(std::ifstream &file, size_t offset,
                                   int64_t total_images) {
  std::vector<size_t> num_objects;
  num_objects.resize(total_images);

  file.clear();
  file.seekg(offset);
  file.read(reinterpret_cast<char *>(num_objects.data()),
            total_images * sizeof(size_t));

  if (file.eof()) LOG(ERROR) << "Reached end of stream";
  if (file.fail()) throw std::runtime_error("Failed reading file.");
  return num_objects;
}

template <typename T>
class TensorReader {
 public:
  TensorReader(std::ifstream &file, size_t beginning_offset, std::string name)
      : file_(file), position_(beginning_offset), name_(name) {}

  PaddleTensor NextBatch(std::vector<int> shape, std::vector<size_t> lod) {
    int numel =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    PaddleTensor tensor;
    tensor.name = name_;
    tensor.shape = shape;
    tensor.dtype = GetPaddleDType<T>();
    tensor.data.Resize(numel * sizeof(T));
    if (lod.empty() == false) {
      tensor.lod.clear();
      tensor.lod.push_back(lod);
    }
    file_.seekg(position_);
    file_.read(reinterpret_cast<char *>(tensor.data.data()), numel * sizeof(T));
    position_ = file_.tellg();
    if (file_.eof()) LOG(ERROR) << name_ << ": reached end of stream";
    if (file_.fail())
      throw std::runtime_error(name_ + ": failed reading file.");
    return tensor;
  }

 protected:
  std::ifstream &file_;
  size_t position_;
  std::string name_;
};

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs,
              int32_t batch_size = FLAGS_batch_size) {
  std::ifstream file(FLAGS_infer_data, std::ios::binary);
  if (!file) {
    FAIL() << "Couldn't open file: " << FLAGS_infer_data;
  }

  int64_t total_images{0};
  file.read(reinterpret_cast<char *>(&total_images), sizeof(int64_t));
  LOG(INFO) << "Total images in file: " << total_images;

  size_t image_beginning_offset = static_cast<size_t>(file.tellg());
  auto lod_offset_in_file =
      image_beginning_offset + sizeof(float) * total_images * 3 * 300 * 300;
  auto labels_beginning_offset =
      lod_offset_in_file + sizeof(size_t) * total_images;

  std::vector<size_t> lod_full =
      ReadObjectsNum(file, lod_offset_in_file, total_images);
  size_t sum_objects_num =
      std::accumulate(lod_full.begin(), lod_full.end(), 0UL);

  auto bbox_beginning_offset =
      labels_beginning_offset + sizeof(int64_t) * sum_objects_num;
  auto difficult_beginning_offset =
      bbox_beginning_offset + sizeof(float) * sum_objects_num * 4;

  TensorReader<float> image_reader(file, image_beginning_offset, "image");
  TensorReader<int64_t> label_reader(file, labels_beginning_offset, "gt_label");
  TensorReader<float> bbox_reader(file, bbox_beginning_offset, "gt_bbox");
  TensorReader<int64_t> difficult_reader(file, difficult_beginning_offset,
                                         "gt_difficult");
  auto iterations_max = total_images / batch_size;
  auto iterations = iterations_max;
  if (FLAGS_iterations > 0 && FLAGS_iterations < iterations_max) {
    iterations = FLAGS_iterations;
  }
  for (auto i = 0; i < iterations; i++) {
    auto images_tensor = image_reader.NextBatch({batch_size, 3, 300, 300}, {});
    std::vector<size_t> batch_lod(lod_full.begin() + i * batch_size,
                                  lod_full.begin() + batch_size * (i + 1));
    size_t batch_num_objects =
        std::accumulate(batch_lod.begin(), batch_lod.end(), 0UL);
    batch_lod.insert(batch_lod.begin(), 0UL);
    for (auto it = batch_lod.begin() + 1; it != batch_lod.end(); it++) {
      *it = *it + *(it - 1);
    }
    auto labels_tensor = label_reader.NextBatch(
        {static_cast<int>(batch_num_objects), 1}, batch_lod);
    auto bbox_tensor = bbox_reader.NextBatch(
        {static_cast<int>(batch_num_objects), 4}, batch_lod);
    auto difficult_tensor = difficult_reader.NextBatch(
        {static_cast<int>(batch_num_objects), 1}, batch_lod);

    inputs->emplace_back(std::vector<PaddleTensor>{
        std::move(images_tensor), std::move(bbox_tensor),
        std::move(labels_tensor), std::move(difficult_tensor)});
  }
}

TEST(Analyzer_bfloat16_mobilenet_ssd, bfloat16) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  AnalysisConfig q_cfg;
  SetConfig(&q_cfg);

  // read data from file and prepare batches with test data
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);

  // configure quantizer
  q_cfg.EnableMkldnnBFloat16();

  // 0 is avg_cost, 1 is top1_acc, 2 is top5_acc or mAP
  CompareBFloat16AndAnalysis(&cfg, &q_cfg, input_slots_all, 2);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
