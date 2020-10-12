// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/math/topk_grad.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    TopKGrad,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    TopKGrad<float>);

namespace {
template <typename T>
void SetTopKGradient(
    const T* values,
    const int64_t* indices,
    const int k,
    const int64_t src_offset,
    const int64_t dst_offset,
    const int64_t stride,
    T* gradient) {
  int64_t src_pos = src_offset;
  for (int i = 0; i < k; ++i) {
    if (indices[src_pos] < 0) {
      continue;
    }
    gradient[dst_offset + indices[src_pos] * stride] = values[src_pos];
    src_pos += stride;
  }
}
} // namespace

template <typename T>
Status TopKGrad<T>::ComputeImpl(const Tensor& indices, const Tensor& grad, Tensor& output) const {
  const int64_t* indices_data = indices.template Data<int64_t>();
  const T* grad_data = grad.template Data<T>();
  T* output_data = output.template MutableData<T>();

  const TensorShape& data_shape = output.Shape();
  const TensorShape& grad_shape = grad.Shape();
  const int64_t axis = HandleNegativeAxis(axis_, data_shape.NumDimensions());
  const int k = static_cast<int>(grad_shape[axis]);
  data_shape.SizeFromDimension(axis);
  const int64_t prev_size = grad_shape.SizeToDimension(axis);
  const int64_t next_size = grad_shape.SizeFromDimension(axis + 1);
  const int64_t src_offset_stride = k * next_size;
  const int64_t dst_offset_stride = data_shape[axis] * next_size;
  int64_t src_offset = 0;
  int64_t dst_offset = 0;
  for (int64_t i = 0; i < prev_size; ++i) {
    for (int64_t j = 0; j < next_size; ++j) {
      SetTopKGradient(
          grad_data,
          indices_data,
          k,
          src_offset + j,
          dst_offset + j,
          next_size,
          output_data);
    }

    src_offset += src_offset_stride;
    dst_offset += dst_offset_stride;
  }

  return Status::OK();
}

template <typename T>
Status TopKGrad<T>::Compute(OpKernelContext* context) const {
  const auto* dY = context->Input<Tensor>(0);
  ORT_ENFORCE(dY);
  const auto* Y1 = context->Input<Tensor>(1);
  ORT_ENFORCE(Y1);
  const auto* X = context->Input<Tensor>(2);
  ORT_ENFORCE(X);
  Tensor* dX = context->Output(0, X->Shape());
  ORT_ENFORCE(dX);
  memset(dX->MutableDataRaw(), 0, dX->SizeInBytes());
  return ComputeImpl(*Y1, *dY, *dX);
}

}  // namespace contrib
}  // namespace onnxruntime