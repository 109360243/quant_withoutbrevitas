import torch
from brevitas.nn import QuantIdentity
from common import Int8ActPerTensorFloatScratch

quant_identity = QuantIdentity(
        act_quant=Int8ActPerTensorFloatScratch,
        return_quant_tensor=True)

float_input = torch.tensor([-1.2, -0.5, 0, 0.3, 0.8, 1.5, 0.8])
print(float_input)
int_output = quant_identity(float_input).int()
print(int_output)
