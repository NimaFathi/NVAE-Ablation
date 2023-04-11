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
            dist.all# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This file has been modified from a file in the PyTorch library.
#
# Source:
# https://github.com/pytorch/pytorch/blob/881c1adfcd916b6cd5de91bc343eb86aff88cc80/torch/nn/modules/batchnorm.py
#
# The license for the original version of this file can be
# found in this directory (LICENSE_PyTorch). The modifications
# to this file are subject to the NVIDIA Source Code License for
# NVAE located at the root directory.
# ---------------------------------------------------------------

from __future__ import division

import torch
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F

from .functions import SyncBatchNorm as sync_batch_norm
from .swish import Swish as swish


class SyncBatchNormSwish(_BatchNorm):
    r"""Applies Batch Normalization over a N-Dimensional input (a mini-batch of [N-2]D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`_ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over all
    mini-batches of the same process groups. :math:`\gamma` and :math:`\beta`
    are learnable parameter vectors of size `C` (where `C` is the input size).
    By default, the elements of :math:`\gamma` are sampled from
    :math:`\mathcal{U}(0, 1)` and the elements of :math:`\beta` are set to 0.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momemtum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, +)` slices, it's common terminology to call this Volumetric Batch Normalization
    or Spatio-temporal Batch Normalization.

    Currently SyncBatchNorm only supports DistributedDataParallel with single GPU per process. Use
    torch.nn.SyncBatchNorm.convert_sync_batchnorm() to convert BatchNorm layer to SyncBatchNorm before wrapping
    Network with DDP.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, +)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics and always uses batch
            statistics in both training and eval modes. Default: ``True``
        process_group: synchronization of stats happen within each process group
            individually. Default behavior is synchronization across the whole
            world

    Shape:
        - Input: :math:`(N, C, +)`
        - Output: :math:`(N, C, +)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.SyncBatchNorm(100)
        >>> # creating process group (optional)
        >>> # process_ids is a list of int identifying rank ids.
        >>> process_group = torch.distributed.new_group(process_ids)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False, process_group=process_group)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)

        >>> # network is nn.BatchNorm layer
        >>> sync_bn_network = nn.SyncBatchNorm.convert_sync_batchnorm(network, process_group)
        >>> # only single gpu per process is currently supported
        >>> ddp_sync_bn_network = torch.nn.parallel.DistributedDataParallel(
        >>>                         sync_bn_network,
        >>>                         device_ids=[args.local_rank],
        >>>                         output_device=args.local_rank)

    .. _`Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift`:
        https://arxiv.org/abs/1502.03167
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, process_group=None):
        super(SyncBatchNormSwish, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        self.process_group = process_group
        # gpu_size is set through DistributedDataParallel initialization. This is to ensure that SyncBatchNorm is used
        # under supported condition (single GPU per process)
        self.ddp_gpu_size = None

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError('expected at least 2D input (got {}D input)'
                             .format(input.dim()))

    def _specify_ddp_gpu_num(self, gpu_size):
        if gpu_size > 1:
            raise ValueError('SyncBatchNorm is only supported for DDP with single GPU per process')
        self.ddp_gpu_size = gpu_size

    def forward(self, input):
        # currently only GPU input is supported
        if not input.is_cuda:
            raise ValueError('SyncBatchNorm expected input tensor to be on GPU')

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        need_sync = self.training or not self.track_running_stats
        if need_sync:
            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            out = F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
            return swish.apply(out)
        else:
            # av: I only use it in this setting.
            if not self.ddp_gpu_size and False:
                raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')

            return sync_batch_norm.apply(
                input, self.weight, self.bias, self.running_mean, self.running_var,
                self.eps, exponential_average_factor, process_group, world_size)_gather(combined_list, combined, process_group, async_op=False)
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