import os
import sys

import torch
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load, _import_module_from_library

# Add CUDA DLL directory for Windows
if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
    cuda_path = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_path:
        cuda_bin = os.path.join(cuda_path, 'bin')
        if os.path.exists(cuda_bin):
            os.add_dll_directory(cuda_bin)

module_path = os.path.dirname(__file__)
USE_FUSED_EXT = True
try:
    fused = load(
        'fused',
        sources=[
            os.path.join(module_path, 'fused_bias_act.cpp'),
            os.path.join(module_path, 'fused_bias_act_kernel.cu'),
        ],
        extra_cuda_cflags=['-allow-unsupported-compiler', '-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'],
        extra_cflags=['-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH'],
    )
except Exception as _e:
    # Fallback: compiled extension not available (e.g., no MSVC). Use a pure-PyTorch implementation.
    USE_FUSED_EXT = False
    class _FusedFallback:
        @staticmethod
        def fused_bias_act(x, bias, out, mode, device, negative_slope, scale):
            # mode/device args are ignored in fallback.
            if bias is None or bias.numel() == 0:
                y = x
            else:
                # bias shape is (C,)
                b = bias.view(1, -1, *([1] * (x.dim() - 2)))
                y = x + b
            # apply leaky relu for fused leaky_relu behavior
            return torch.nn.functional.leaky_relu(y, negative_slope) * scale

    fused = _FusedFallback()

#fused = _import_module_from_library('fused', '/tmp/torch_extensions/fused', True)


class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        if USE_FUSED_EXT:
            empty = grad_output.new_empty(0)
            grad_input = fused.fused_bias_act(
                grad_output, empty, out, 3, 1, negative_slope, scale
            )
        else:
            # approximate gradient by autograd (not used since fallback uses autograd)
            grad_input = grad_output

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        grad_bias = grad_input.sum(dim).detach()

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(
            gradgrad_input, gradgrad_bias, out, 3, 1, ctx.negative_slope, ctx.scale
        )

        return gradgrad_out, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        if USE_FUSED_EXT:
            empty = input.new_empty(0)
            out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
            ctx.save_for_backward(out)
            ctx.negative_slope = negative_slope
            ctx.scale = scale
            return out
        else:
            # Pure PyTorch fallback: add bias then LeakyReLU * scale
            if bias is None or bias.numel() == 0:
                y = input
            else:
                b = bias.view(1, -1, *([1] * (input.dim() - 2)))
                y = input + b
            out = torch.nn.functional.leaky_relu(y, negative_slope) * scale
            ctx.save_for_backward(y)
            ctx.negative_slope = negative_slope
            ctx.scale = scale
            return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        if USE_FUSED_EXT:
            grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
                grad_output, out, ctx.negative_slope, ctx.scale
            )
            return grad_input, grad_bias, None, None
        else:
            # Let autograd handle gradients for fallback
            y = out
            grad_y = grad_output * ctx.scale
            grad_input = torch.where(y > 0, grad_y, grad_y * ctx.negative_slope)
            # approximate grad_bias by summing over dimensions
            dim = [0]
            if grad_input.ndim > 2:
                dim += list(range(2, grad_input.ndim))
            grad_bias = grad_input.sum(dim)
            return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
