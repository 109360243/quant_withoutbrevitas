import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint  # 不再使用 Int8ActPerTensorFloat
from test5 import RefinedNumPyDecoupledRescalingIntQuantWithInput
class QuantizedMLP(nn.Module):
    def __init__(self):
        super(QuantizedMLP, self).__init__()
        
        # Input Layer (784 -> 256), 假設是用於 MNIST (28x28 = 784)
        self.fc1 = qnn.QuantLinear(
            in_features=784, 
            out_features=256, 
            weight_quant=RefinedNumPyDecoupledRescalingIntQuantWithInput, 
            bias=True)

        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平成 1D 向量
        x = (self.fc1(x))
        return x

# 測試一下 MLP
model = QuantizedMLP()
dummy_input = torch.randn(1, 1, 28, 28)  # MNIST 標準尺寸
output = model(dummy_input)
print(output)