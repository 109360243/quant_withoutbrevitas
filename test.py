import numpy as np

class IntQuant_np():
    """
    Implements scale, shifted, uniform integer quantization of an input tensor,
    according to an input scale, zero-point, and bit-width.

    Args:
        narrow_range (bool): Restrict quantization to a narrow range or not.
        signed (bool): Quantize to a signed range or not.

    Returns:
        np.array: Quantized output in de-quantized format.
    """

    __constants__ = ['signed', 'narrow_range']

    def __init__(self, narrow_range: bool, signed: bool):
        self.signed = signed
        self.narrow_range = narrow_range
      
    def tensor_clamp(self, x: np.array, min_val: np.array, max_val: np.array) -> np.array:
        out = np.where(x > max_val, max_val, x)
        out = np.where(out < min_val, min_val, out)
        return out

    def to_int(self, scale: np.array, zero_point: np.array, bit_width: np.array, x: np.array) -> np.array:
        y = x / scale
        y = y + zero_point
        min_int_val = self.min_int(bit_width)
        max_int_val = self.max_int(bit_width)
        y = np.round(y)
        y = self.tensor_clamp(y, min_val=min_int_val, max_val=max_int_val)
        return y

    def min_int(self, bit_width: np.array) -> np.array:
        if self.signed and self.narrow_range:
            return -(2 ** (bit_width - 1)) + 1
        elif self.signed and not self.narrow_range:
            return -(2 ** (bit_width - 1))
        else:
            return 0 * bit_width

    def max_int(self, bit_width: np.array) -> np.array:
        if not self.signed and not self.narrow_range:
            return (2 ** bit_width) - 1
        elif not self.signed and self.narrow_range:
            return (2 ** bit_width) - 2
        else:
            return (2 ** (bit_width - 1)) - 1

    def forward(self, scale: np.array, zero_point: np.array, bit_width: np.array, x: np.array) -> np.array:
        y_int = self.to_int(scale, zero_point, bit_width, x)
        y = (y_int - zero_point) * scale
        return y


class RescalingIntQuant():
    """
    A wrapper for integer quantization. It scales, sets zero-point and bit-width for quantization.

    Returns:
        Tuple[np.array, np.array, np.array, np.array]: Quantized output, scale, zero-point, and bit-width.
    """

    def __init__(self):
        self.int_quant = IntQuant_np(narrow_range=True, signed=True)

    def forward(self, x: np.array) -> tuple:
        bit_width = np.array(float(4))
        threshold = np.array(0.1000)
        int_threshold = -min_int(signed  = True, narrow_range = True,bit_width = np.array(4))
        scale = threshold / int_threshold
        zero_point = np.array(0.0)
        y = self.int_quant.forward(scale, zero_point, bit_width, x)
        return y, scale, zero_point, bit_width
def min_int(signed: bool, narrow_range: bool, bit_width: np.array) -> np.array:
    """ Compute the minimum integer representable by a given number of bits.

    Args:
        signed (bool): Indicates whether the represented integer is signed or not.
        narrow_range (bool): Indicates whether to narrow the minimum value represented by 1.
        bit_width (Tensor): Number of bits available for the representation.


    Returns:
        Tensor: Maximum unsigned integer that can be represented according to the input arguments.

    Examples:
        >>> min_int(signed=True, narrow_range=True, bit_width=torch.tensor(8))
        tensor(-127)
        >>> min_int(signed=False, narrow_range=True, bit_width=torch.tensor(8))
        tensor(0)
        >>> min_int(signed=True, narrow_range=False, bit_width=torch.tensor(8))
        tensor(-128)
        >>> min_int(signed=False, narrow_range=False, bit_width=torch.tensor(8))
        tensor(0)
    """
    if signed and narrow_range:
        value = -(2 ** (bit_width - 1)) + 1
    elif signed and not narrow_range:
        value = -(2 ** (bit_width - 1))
    else:
        value = 0 * bit_width
    return value
    

rescaling_int_quant = RescalingIntQuant()
input_data = np.array(([-1.2, -0.5, 0, 0.3, 0.8, 1.5, 0.8]))
output, scale, zero_point, bit_width = rescaling_int_quant.forward(input_data)

print("Output:", (output))
print("output.type",(output).dtype)
print("Scale:", scale)
print("Zero Point:", zero_point) 
print("Bit Width:", bit_width)