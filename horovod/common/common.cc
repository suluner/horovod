// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <sstream>
#include <cassert>

#include "common.h"

namespace horovod {
namespace common {

Status::Status() = default;

Status::Status(StatusType type, std::string reason) {
  type_ = type;
  reason_ = reason;
}

Status Status::OK() {
  return Status();
}

Status Status::UnknownError(std::string message) {
  return Status(StatusType::UNKNOWN_ERROR, message);
}

Status Status::PreconditionError(std::string message) {
  return Status(StatusType::PRECONDITION_ERROR, message);
}

Status Status::Aborted(std::string message) {
  return Status(StatusType::ABORTED, message);
}

Status Status::InvalidArgument(std::string message) {
  return Status(StatusType::INVALID_ARGUMENT, message);
}

Status Status::InProgress() {
  return Status(StatusType::IN_PROGRESS, "");
}

bool Status::ok() const {
  return type_ == StatusType::OK;
}

bool Status::in_progress() const {
  return type_ == StatusType::IN_PROGRESS;
}

StatusType Status::type() const {
  return type_;
}

const std::string& Status::reason() const {
  return reason_;
}

void TensorShape::AddDim(int64_t dim) {
  shape_.push_back(dim);
}

void TensorShape::AppendShape(TensorShape& other) {
  for (auto dim : other.shape_) {
    shape_.push_back(dim);
  }
}

const std::string TensorShape::DebugString() const {
  std::stringstream args;
  args << "[";
  for (auto it = shape_.begin(); it != shape_.end(); ++it) {
    if (it != shape_.begin()) {
      args << ", ";
    }
    args << *it;
  }
  args << "]";
  return args.str();
}

int TensorShape::dims() const {
  return (int)shape_.size();
}

int64_t TensorShape::dim_size(int idx) const {
  assert(idx >= 0);
  assert(idx < shape_.size());
  return shape_[idx];
}

int64_t TensorShape::num_elements() const {
  int64_t result = 1;
  for (auto dim : shape_) {
    result *= dim;
  }
  return result;
}

const std::vector<int64_t>& TensorShape::to_vector() const { return shape_; }

template <DataType DT, class T>
DummyTensor<DT, T>::DummyTensor(int device, int64_t num_elements) {
  num_elements_ = num_elements;
  device_ = device;
  if (device_ == CPU_DEVICE_ID) {
    buffer_data_ = new T[num_elements_];
    memset(buffer_data_, 0, sizeof(T)*num_elements_);
  } else {
    #if HAVE_CUDA
      cudaSetDevice(device_);
      cudaMalloc(&buffer_data_, sizeof(T)*num_elements_);
      cudaMemset(buffer_data_, 0, sizeof(T)*num_elements_);
    #else
      throw std::logic_error("Internal error. Requested Join "
                             "with GPU device but not compiled with CUDA.");
    #endif
 }
}

template <DataType DT, class T>
DummyTensor<DT, T>::~DummyTensor() {
  if (device_ == CPU_DEVICE_ID) {
    delete[] buffer_data_;
  } else {
    #if HAVE_CUDA
      cudaFree(buffer_data_);
    #else
      throw std::logic_error("Internal error. Requested Join "
                             "with GPU device but not compiled with CUDA.");
    #endif
  }
}

template <DataType DT, class T>
const DataType DummyTensor<DT, T>::dtype() const {
  return DT;
}

template <DataType DT, class T>
const TensorShape DummyTensor<DT, T>::shape() const {
  TensorShape shape;
  shape.AddDim(num_elements_);
  return shape;
}

template <DataType DT, class T>
const void* DummyTensor<DT, T>::data() const {
  return (void*) buffer_data_;
}

template <DataType DT, class T>
int64_t DummyTensor<DT, T>::size() const {
  return num_elements_*sizeof(T);
}

} // namespace common
} // namespace horovod
