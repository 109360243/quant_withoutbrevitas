import numpy as np
from typing import Optional, Tuple

class SymmetricNumpyInt8Quantizer:
    def __init__(self, collect_stats_steps: int = 300):
        self.collect_stats_steps = collect_stats_steps
        self.step_count = 0
        self.scale = None
        self.stats_buffer = []

    def _collect_stats(self, x: np.ndarray):
        # Collect statistics over multiple steps
        self.stats_buffer.append(np.abs(x).flatten())
        self.step_count += 1

        if self.step_count == self.collect_stats_steps:
            # Find the maximum absolute value in the collected statistics
            all_stats = np.concatenate(self.stats_buffer)
            threshold = np.max(all_stats)  # Use max absolute value
            self.scale = threshold / 127  # Use 127 for 8-bit signed int quantization
            self.stats_buffer = []  # Clear buffer to save memory

    def quantize(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.step_count < self.collect_stats_steps:
            self._collect_stats(x)
            return x, x  # During stats collection, return the original tensor

        if self.scale is None:
            raise ValueError("Statistics have not been collected yet.")

        # Apply symmetric quantization
        x_scaled = x / self.scale
        x_clipped = np.clip(x_scaled, -127, 127)  # Ensure the range is [-127, 127]
        x_int8 = np.round(x_clipped).astype(np.int8)

        # Dequantization for comparison
        x_dequantized = x_int8 * self.scale

        return x_int8, x_dequantized

    def get_scale(self) -> Optional[float]:
        return self.scale

# Example usage
if __name__ == "__main__":
    quantizer = SymmetricNumpyInt8Quantizer()

    # Simulate 300 steps of data
    for _ in range(300):
        data = np.random.randn(1000).astype(np.float32)
        _, _ = quantizer.quantize(data)

    # After 300 steps, quantizer starts actual quantization
    test_data = np.array([-1.2, -0.5, 0, 0.3, 0.8, 1.5, 0.8]).astype(np.float32)
    quantized_int8, quantized_float = quantizer.quantize(test_data)

    print(f"Original data range: [{test_data.min():.4f}, {test_data.max():.4f}]")
    print(f"Quantized int8 data range: [{quantized_int8.min()}, {quantized_int8.max()}]")
    print(f"Dequantized float data range: [{quantized_float.min():.4f}, {quantized_float.max():.4f}]")
    print(f"Quantization scale: {quantizer.get_scale():.6f}")

    # Verification of symmetry
    zero_indices = np.abs(test_data) < 1e-6
    if zero_indices.any():
        first_zero_index = zero_indices.nonzero()[0][0]
        print(f"Quantized zero: {quantized_int8[first_zero_index]}")
    else:
        print("No elements in test_data are close to zero.")

    print(f"Max positive value: {quantized_int8.max()}")  # Should be 127
    print(f"Min negative value: {quantized_int8.min()}")  # Should be -127
    print(quantized_int8)
