from brevitas.core.scaling import ConstScaling
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.core.scaling import IntScaling
from brevitas.core.quant import IntQuant
from brevitas.core.bit_width import BitWidthConst
import torch
from torch import Tensor
from torch.nn import Module
from typing import Optional, Tuple


class RescalingIntQuant():
    """
    ScriptModule that wraps around an integer quantization implementation like
    :class:`~brevitas.core.quant.IntQuant`. Scale, zero-point and bit-width are returned from their
    respective implementations and passed on to the integer quantization implementation.

    Args:
        int_quant (Module): Module that implements integer quantization.
        scaling_impl (Module): Module that takes in the input to quantize and returns a scale factor,
            here interpreted as threshold on the floating-point range of quantization.
        int_scaling_impl (Module): Module that takes in a bit-width and returns an integer scale
            factor, here interpreted as threshold on the integer range of quantization.
        zero_point_impl (Module): Module that returns an integer zero-point.
        bit_width_impl (Module): Module that returns a bit-width.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Quantized output in de-quantized format, scale,
            zero-point, bit_width.

    Examples:
        >>> from brevitas.core.scaling import ConstScaling
        >>> from brevitas.core.zero_point import ZeroZeroPoint
        >>> from brevitas.core.scaling import IntScaling
        >>> from brevitas.core.quant import IntQuant
        >>> from brevitas.core.bit_width import BitWidthConst
        >>> int_quant_wrapper = RescalingIntQuant(
        ...                         IntQuant(narrow_range=True, signed=True),
        ...                         ConstScaling(0.1),
        ...                         IntScaling(signed=True, narrow_range=True),
        ...                         ZeroZeroPoint(),
        ...                         BitWidthConst(4))
        >>> inp = torch.Tensor([0.042, -0.053, 0.31, -0.44])
        >>> out, scale, zero_point, bit_width = int_quant_wrapper(inp)
        >>> out
        tensor([ 0.0429, -0.0571,  0.1000, -0.1000])
        >>> scale
        tensor(0.0143)
        >>> zero_point
        tensor(0.)
        >>> bit_width
        tensor(4.)

    Note:
        scale = scaling_impl(x) / int_scaling_impl(bit_width)

    Note:
        Set env variable BREVITAS_JIT=1 to enable TorchScript compilation of this module.
    """

    def __init__(
            self,
            int_quant: Module,
            scaling_impl: Module,
            int_scaling_impl: Module,
            zero_point_impl: Module,
            bit_width_impl: Module):
        super(RescalingIntQuant, self).__init__()
        self.int_quant = int_quant
        self.scaling_impl = scaling_impl
        self.int_scaling_impl = int_scaling_impl
        self.zero_point_impl = zero_point_impl
        self.msb_clamp_bit_width_impl = bit_width_impl

    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        bit_width = self.msb_clamp_bit_width_impl()
        threshold = self.scaling_impl(x)
        int_threshold = self.int_scaling_impl(bit_width)
        scale = threshold / int_threshold
        zero_point = self.zero_point_impl(x, scale, bit_width)
        y = self.int_quant(scale, zero_point, bit_width, x)
        return y, scale, zero_point, bit_width
    
inp = torch.Tensor([0.042, -0.053, 0.31, -0.44])
int_quant_wrapper = RescalingIntQuant(
                        IntQuant(narrow_range=True, signed=True),
                        ConstScaling(0.1),
                        IntScaling(signed=True, narrow_range=True),
                        ZeroZeroPoint(),
                        BitWidthConst(4))

out, scale, zero_point, bit_width = int_quant_wrapper.forward(inp)
print("out:",out.type(),"scale:",scale)


#鎖定 scale 問題