import numpy as np

class RefinedNumPyDecoupledRescalingIntQuantWithInput:
    def init(self):
        pass

    def msbclampbitwidthimpl(self):
        return 8

    def int_scaling_impl(self, bit_width):
        return 2(bit_width - 1) - 1

    def scaling_impl(self, x):

        return np.max(np.abs(x)) / 127  # 127 是 8 位有符號整數的最大值

    def zero_point_impl(self, x, scale, bit_width):
        return 0

    def decoupled_int_quant(self, scale, zero_point, bit_width, x):
        x_scaled = x / scale
        x_rounded = np.round(x_scaled)
        x_clipped = np.clip(x_rounded, -2(bit_width-1), 2*(bit_width-1)-1)
        return x_clipped.astype(int)

    def forward(self, x, input_bit_width, input_is_signed):
        bit_width = self.msbclampbitwidthimpl()
        scale = self.scaling_impl(x)
        zero_point = self.zero_point_impl(x, scale, bit_width)
        y_int = self.decoupled_int_quant(scale, zero_point, bit_width, x)
        y = y_int*scale  # 縮放回浮點範圍
        return y, scale, zero_point, bit_width, y_int


if __name__ == "__main__":
    x = np.array([-1.2, -0.5, 0, 0.3, 0.8, 1.5, 0.8])
    input_bit_width = np.array([8])
    input_is_signed = True

    quantizer = RefinedNumPyDecoupledRescalingIntQuantWithInput()
    y, scale, zero_point, bit_width, y_int = quantizer.forward(x, input_bit_width, input_is_signed)

    print("Original input:", x)
    print("Quantized output (float):", y)
    print("Quantized output (int):", y_int)
    print("Scale:", scale)
    print("Zero point:", zero_point)
    print("Bit width:", bit_width)
    print("Mean absolute quantization error:", np.mean(np.abs(x - y)))


    print("\nVerification:")
    print("Float from int:", y_int * scale)
    print("Int from float:", np.round(y / scale).astype(int))