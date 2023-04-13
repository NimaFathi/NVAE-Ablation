# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the PyTorch library.
#
# Source:
# https://github.com/pytorch/pytorch/blob/2a54533c64c409b626b6c209ed78258f67aec194/torch/nn/modules/_functions.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_PyTorch). The modifications
# to this file are subject to the NVIDIA Source Code License for
# NVAE located at the root directory.
# ---------------------------------------------------------------

import torch
from torch.autograd.function import Function
import torch.distributed as dist


# class SyncBatchNorm(Function):

#     @staticmethod
#     def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
#         if not (
#             input.is_contiguous(memory_format=torch.channels_last) or
#             input.is_contiguous(memory_format=torch.channels_last_3d)
#         ):
#             input = input.contiguous()
#         if weight is not None:
#             weight = weight.contiguous()

#         size = int(input.numel() // input.size(1))
#         if size == 1 and world_size < 2:
#             raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))

#         num_channels = input.shape[1]
#         if input.numel() > 0:
#             # calculate mean/invstd for input.
#             mean, invstd = torch.batch_norm_stats(input, eps)

#             count = torch.full(
#                 (1,),
#                 input.numel() // input.size(1),
#                 dtype=mean.dtype,
#                 device=mean.device
#             )

#             # C, C, 1 -> (2C + 1)
#             combined = torch.cat([mean, invstd, count], dim=0)
#         else:
#             # for empty input, set stats and the count to zero. The stats with
#             # zero count will be filtered out later when computing global mean
#             # & invstd, but they still needs to participate the all_gather
#             # collective communication to unblock other peer processes.
#             combined = torch.zeros(
#                 2 * num_channels + 1,
#                 dtype=input.dtype,
#                 device=input.device
#             )

#         # Use allgather instead of allreduce because count could be different across
#         # ranks, simple all reduce op can not give correct results.
#         # batch_norm_gather_stats_with_counts calculates global mean & invstd based on
#         # all gathered mean, invstd and count.
#         # for nccl backend, use the optimized version of all gather.
#         if process_group._get_backend_name() == 'nccl':
#             # world_size * (2C + 1)
#             combined_size = combined.numel()
#             combined_flat = torch.empty(1,
#                                         combined_size * world_size,
#                                         dtype=combined.dtype,
#                                         device=combined.device)
#             dist.all_gather_into_tensor(combined_flat, combined, process_group, async_op=False)
#             combined = torch.reshape(combined_flat, (world_size, combined_size))
#             # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
#             mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
#         else:
#             # world_size * (2C + 1)
#             combined_list = [
#                 torch.empty_like(combined) for _ in range(world_size)
#             ]
#             dist.all_gather(combined_list, combined, process_group, async_op=False)
#             combined = torch.stack(combined_list, dim=0)
#             # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
#             mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)

#         if not torch.cuda.is_current_stream_capturing():
#             # The lines below force a synchronization between CUDA and CPU, because
#             # the shape of the result count_all depends on the values in mask tensor.
#             # Such synchronizations break CUDA Graph capturing.
#             # See https://github.com/pytorch/pytorch/issues/78549
#             # FIXME: https://github.com/pytorch/pytorch/issues/78656 describes
#             # a better longer-term solution.

#             # remove stats from empty inputs
#             mask = count_all.squeeze(-1) >= 1
#             count_all = count_all[mask]
#             mean_all = mean_all[mask]
#             invstd_all = invstd_all[mask]

#         # calculate global mean & invstd
#         mean, invstd = torch.batch_norm_gather_stats_with_counts(
#             input,
#             mean_all,
#             invstd_all,
#             running_mean,
#             running_var,
#             momentum,
#             eps,
#             count_all.view(-1)
#         )

#         self.save_for_backward(input, weight, mean, invstd, count_all.to(torch.int32))
#         self.process_group = process_group

#         # apply element-wise normalization
#         if input.numel() > 0:
#             return torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
#         else:
#             return torch.empty_like(input)

#     @staticmethod
#     def backward(self, grad_output):
#         if not (
#             grad_output.is_contiguous(memory_format=torch.channels_last) or
#             grad_output.is_contiguous(memory_format=torch.channels_last_3d)
#         ):
#             grad_output = grad_output.contiguous()
#         saved_input, weight, mean, invstd, count_tensor = self.saved_tensors
#         grad_input = grad_weight = grad_bias = None
#         process_group = self.process_group

#         if saved_input.numel() > 0:
#             # calculate local stats as well as grad_weight / grad_bias
#             sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
#                 grad_output,
#                 saved_input,
#                 mean,
#                 invstd,
#                 weight,
#                 self.needs_input_grad[0],
#                 self.needs_input_grad[1],
#                 self.needs_input_grad[2]
#             )

#             if self.needs_input_grad[0]:
#                 # synchronizing stats used to calculate input gradient.
#                 num_channels = sum_dy.shape[0]
#                 combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
#                 torch.distributed.all_reduce(
#                     combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
#                 sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

#                 # backward pass for gradient calculation
#                 grad_input = torch.batch_norm_backward_elemt(
#                     grad_output,
#                     saved_input,
#                     mean,
#                     invstd,
#                     weight,
#                     sum_dy,
#                     sum_dy_xmu,
#                     count_tensor
#                 )
#             # synchronizing of grad_weight / grad_bias is not needed as distributed
#             # training would handle all reduce.
#             if weight is None or not self.needs_input_grad[1]:
#                 grad_weight = None

#             if weight is None or not self.needs_input_grad[2]:
#                 grad_bias = None
#         else:
#             # This process got an empty input tensor in the forward pass.
#             # Although this process can directly set grad_input as an empty
#             # tensor of zeros, it still needs to participate in the collective
#             # communication to unblock its peers, as other peer processes might
#             # have recieved non-empty inputs.
#             num_channels = saved_input.shape[1]
#             if self.needs_input_grad[0]:
#                 # launch all_reduce to unblock other peer processes
#                 combined = torch.zeros(
#                     2 * num_channels,
#                     dtype=saved_input.dtype,
#                     device=saved_input.device
#                 )
#                 torch.distributed.all_reduce(
#                     combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)

#             # Leave grad_input, grad_weight and grad_bias as None, which will be
#             # interpreted by the autograd engine as Tensors full of zeros.

#         return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


class SyncBatchNorm(Function):

    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        if not (
            input.is_contiguous(memory_format=torch.channels_last) or
            input.is_contiguous(memory_format=torch.channels_last_3d)
        ):
            input = input.contiguous()
        if weight is not None:
            weight = weight.contiguous()

        size = int(input.numel() // input.size(1))
        if size == 1 and world_size < 2:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))

        num_channels = input.shape[1]
        if input.numel() > 0:
            # calculate mean/invstd for input.
            mean, invstd = torch.batch_norm_stats(input, eps)

            count = torch.full(
                (1,),
                input.numel() // input.size(1),
                dtype=mean.dtype,
                device=mean.device
            )

            # C, C, 1 -> (2C + 1)
            combined = torch.cat([mean, invstd, count], dim=0)
        else:
            # for empty input, set stats and the count to zero. The stats with
            # zero count will be filtered out later when computing global mean
            # & invstd, but they still needs to participate the all_gather
            # collective communication to unblock other peer processes.
            combined = torch.zeros(
                2 * num_channels + 1,
                dtype=input.dtype,
                device=input.device
            )

        # Use allgather instead of allreduce because count could be different across
        # ranks, simple all reduce op can not give correct results.
        # batch_norm_gather_stats_with_counts calculates global mean & invstd based on
        # all gathered mean, invstd and count.
        # for nccl backend, use the optimized version of all gather.
        if process_group._get_backend_name() == 'nccl':
            # world_size * (2C + 1)
            combined_size = combined.numel()
            combined_flat = torch.empty(1,
                                        combined_size * world_size,
                                        dtype=combined.dtype,
                                        device=combined.device)
            dist.all_gather_into_tensor(combined_flat, combined, process_group, async_op=False)
            combined = torch.reshape(combined_flat, (world_size, combined_size))
            # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
            mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
        else:
            # world_size * (2C + 1)
            combined_list = [
                torch.empty_like(combined) for _ in range(world_size)
            ]
            dist.all_gather(combined_list, combined, process_group, async_op=False)
            combined = torch.stack(combined_list, dim=0)
            # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
            mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)

        if not torch.cuda.is_current_stream_capturing():
            # The lines below force a synchronization between CUDA and CPU, because
            # the shape of the result count_all depends on the values in mask tensor.
            # Such synchronizations break CUDA Graph capturing.
            # See https://github.com/pytorch/pytorch/issues/78549
            # FIXME: https://github.com/pytorch/pytorch/issues/78656 describes
            # a better longer-term solution.

            # remove stats from empty inputs
            mask = count_all.squeeze(-1) >= 1
            count_all = count_all[mask]
            mean_all = mean_all[mask]
            invstd_all = invstd_all[mask]

        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            count_all.view(-1)
        )

        self.save_for_backward(input, weight, bias, mean, invstd, count_all.to(torch.int32))
        self.process_group = process_group

        # apply element-wise normalization
        if input.numel() > 0:
            out = torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
            # av: apply swish
            assert eps == 1e-5, "I assumed below that eps is 1e-5"
            out = out * torch.sigmoid(out)
            return out
        else:
            return torch.empty_like(input)

    @staticmethod
    def backward(self, grad_output):
        if not (
            grad_output.is_contiguous(memory_format=torch.channels_last) or
            grad_output.is_contiguous(memory_format=torch.channels_last_3d)
        ):
            grad_output = grad_output.contiguous()
        saved_input, weight, bias, mean, invstd, count_tensor = self.saved_tensors

        # av: re-compute batch normalized out
        eps = 1e-5
        out = torch.batch_norm_elemt(saved_input, weight, bias, mean, invstd, eps)
        sigmoid_out = torch.sigmoid(out)
        grad_output *= (sigmoid_out * (1 + out * (1 - sigmoid_out)))
        # av: end

        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group

        if saved_input.numel() > 0:
            # calculate local stats as well as grad_weight / grad_bias
            sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                self.needs_input_grad[0],
                self.needs_input_grad[1],
                self.needs_input_grad[2]
            )

            if self.needs_input_grad[0]:
                # synchronizing stats used to calculate input gradient.
                num_channels = sum_dy.shape[0]
                combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
                torch.distributed.all_reduce(
                    combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
                sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

                # backward pass for gradient calculation
                grad_input = torch.batch_norm_backward_elemt(
                    grad_output,
                    saved_input,
                    mean,
                    invstd,
                    weight,
                    sum_dy,
                    sum_dy_xmu,
                    count_tensor
                )
            # synchronizing of grad_weight / grad_bias is not needed as distributed
            # training would handle all reduce.
            if weight is None or not self.needs_input_grad[1]:
                grad_weight = None

            if weight is None or not self.needs_input_grad[2]:
                grad_bias = None
        else:
            # This process got an empty input tensor in the forward pass.
            # Although this process can directly set grad_input as an empty
            # tensor of zeros, it still needs to participate in the collective
            # communication to unblock its peers, as other peer processes might
            # have recieved non-empty inputs.
            num_channels = saved_input.shape[1]
            if self.needs_input_grad[0]:
                # launch all_reduce to unblock other peer processes
                combined = torch.zeros(
                    2 * num_channels,
                    dtype=saved_input.dtype,
                    device=saved_input.device
                )
                torch.distributed.all_reduce(
                    combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)

            # Leave grad_input, grad_weight and grad_bias as None, which will be
            # interpreted by the autograd engine as Tensors full of zeros.

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None