import numpy as np

def decoupled_rescaling_int_quant_with_input(x, input_bit_width, input_is_signed):
    # Helper functions
    def msb_clamp_bit_width():
        return 8  # Assuming 8-bit quantization

    def int_scaling(bit_width):
        return 2**(bit_width - 1) - 1

    def pre_scaling(x, input_bit_width, input_is_signed):
        # Simplified pre-scaling, you may need to adjust this
        return np.abs(x).max()

    def zero_point(x, scale, bit_width):
        return 0

    # Main quantization logic
    bit_width = msb_clamp_bit_width()
    int_threshold = int_scaling(bit_width)
    pre_threshold = pre_scaling(x, input_bit_width, input_is_signed)
    pre_scale = pre_threshold / int_threshold
    pre_zero_point = zero_point(x, pre_scale, bit_width)
    
    threshold = np.abs(x).max()
    scale = threshold / int_threshold
    zero_point = zero_point(x, scale, bit_width)
    
    # Quantization
    x_scaled = x / scale + zero_point
    y = np.clip(np.round(x_scaled), -2**(bit_width-1), 2**(bit_width-1)-1).astype(np.int8)
    
    return y, scale, zero_point, bit_width, pre_scale, pre_zero_point

# Example usage
x = np.array([-1.2, -0.5, 0, 0.3, 0.8, 1.5, 0.8])
input_bit_width = np.array([8])  # Assuming float32 input
input_is_signed = True

y, scale, zero_point, bit_width, pre_scale, pre_zero_point = decoupled_rescaling_int_quant_with_input(x, input_bit_width, input_is_signed)
print(y)
print(f"Output shape: {y.shape}")
print(f"Output dtype: {y.dtype}")
print(f"Scale: {scale}")
print(f"Zero point: {zero_point}")
print(f"Bit width: {bit_width}")
print(f"Pre-scale: {pre_scale}")
print(f"Pre-zero point: {pre_zero_point}")