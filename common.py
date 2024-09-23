import torch
from dependencies import value

from brevitas.inject.enum import *
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import FloatToIntImplType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.quant.solver import WeightQuantSolver, ActQuantSolver
from brevitas.quant.base import *
from brevitas.core.function_wrapper.ops_ste import CeilSte

from brevitas.quant.scaled_int import Int8Bias, \
    Int16Bias, \
    Int32Bias

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""
Float Scaling Factor
"""
class Int8WeightPerTensorFloatScratch(WeightQuantSolver):
    quant_type = QuantType.INT # integer quantization
    bit_width_impl_type = BitWidthImplType.CONST # constant bit width
    float_to_int_impl_type = FloatToIntImplType.ROUND # round to nearest
    scaling_impl_type = ScalingImplType.STATS # scale based on statistics
    scaling_stats_op = StatsOp.MAX # scale statistics is the absmax value
    restrict_scaling_type = RestrictValueType.FP # scale factor is a floating point value
    scaling_per_output_channel = False # scale is per tensor
    bit_width = 8 # bit width is 8
    signed = True # quantization range is signed
    narrow_range = True # quantization range is [-127,127] rather than [-128, 127]
    zero_point_impl = ZeroZeroPoint # zero point is 0.

class Int4WeightPerTensorFloatScratch(Int8WeightPerTensorFloatScratch):
    bit_width=4

class Int2WeightPerTensorFloatScratch(Int8WeightPerTensorFloatScratch):
    bit_width=2

class Int8ActPerTensorFloatScratch(ActQuantSolver):
    quant_type = QuantType.INT # integer quantization-----------------------------------------------------------------#SolveActTensorQuantFromEnum
    bit_width_impl_type = auto() # constant bit width-------------------------------------------------#SolveBitWidthImplFromEnum
    float_to_int_impl_type = FloatToIntImplType.ROUND # round to nearest ---------------------------------------------#SolveTensorQuantFloatToIntImplFromEnum 
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS # scale is a parameter initialized from statistics-------#SolveActScalingImplFromEnum
    scaling_stats_op = StatsOp.PERCENTILE # scale statistics is a percentile of the abs value ------------------------#SolveScalingStatsOpFromEnum
    high_percentile_q = 99.999 # percentile is 99.999
    collect_stats_steps = 300  # statistics are collected for 300 forward steps before switching to a learned parameter
    restrict_scaling_type = RestrictValueType.FP # scale is a floating-point value------------------------------------#SolveIntScalingImplFromEnum
    scaling_per_output_channel = False  # scale is per tensor --------------------------------------------------------#SolveActScalingShape
    bit_width = 8  # bit width is 8
    signed = True # quantization range is signed
    narrow_range = False # quantization range is [-128, 127] rather than [-127, 127]
    zero_point_impl = ZeroZeroPoint # zero point is 0.

class Int16ActPerTensorFloatScratch(Int8ActPerTensorFloatScratch):
    bit_width=16

class Int4ActPerTensorFloatScratch(Int8ActPerTensorFloatScratch):
    bit_width=4

class Int2ActPerTensorFloatScratch(Int8ActPerTensorFloatScratch):
    bit_width=2

class Uint8ActPerTensorFloatScratch(Int8ActPerTensorFloatScratch):
    signed = False

class Uint4ActPerTensorFloatScratch(Uint8ActPerTensorFloatScratch):
    bit_width=4

class Uint2ActPerTensorFloatScratch(Uint8ActPerTensorFloatScratch):
    bit_width=2


"""
Power of Two scaling factor(Fixed Point)
"""
class Int8WeightPerTensorFixedPointScratch(Int8WeightPerTensorFloatScratch):
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    bit_width = 8
    restrict_value_float_to_int_impl = CeilSte

class Int4WeightPerTensorFixedPointScratch(Int8WeightPerTensorFixedPointScratch):
    bit_width = 4

class Int2WeightPerTensorFixedPointScratch(Int8WeightPerTensorFixedPointScratch):
    bit_width = 2

class Int8ActPerTensorFixedPointScratch(Int8ActPerTensorFloatScratch):
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    bit_width = 8
    restrict_value_float_to_int_impl = CeilSte

class Int4ActPerTensorFixedPointScratch(Int8ActPerTensorFixedPointScratch):
    bit_width = 4

class Int2ActPerTensorFixedPointScratch(Int8ActPerTensorFixedPointScratch):
    bit_width = 2
    
class Uint8ActPerTensorFixedPointScratch(Int8ActPerTensorFixedPointScratch):
    signed = False

class Uint4ActPerTensorFixedPointScratch(Int8ActPerTensorFixedPointScratch):
    signed = False
    bit_width = 4

class Uint2ActPerTensorFixedPointScratch(Int8ActPerTensorFixedPointScratch):
    signed = False
    bit_width = 2


weight_quantizer = {'int8': Int8WeightPerTensorFloatScratch,
                    'int4': Int4WeightPerTensorFloatScratch,
                    'int2': Int2WeightPerTensorFloatScratch,
                    'fxp8': Int8WeightPerTensorFixedPointScratch,
                    'fxp4': Int4WeightPerTensorFixedPointScratch,
                    'fxp2': Int2WeightPerTensorFixedPointScratch}

act_quantizer = {
                'int16': Int16ActPerTensorFloatScratch,
                'int8': Int8ActPerTensorFloatScratch,
                'int4': Int4ActPerTensorFloatScratch,
                'int2': Int2ActPerTensorFloatScratch,
                'uint8': Uint8ActPerTensorFloatScratch,
                'uint4': Uint4ActPerTensorFloatScratch,
                'uint2': Uint2ActPerTensorFloatScratch,
                'fxp8': Int8ActPerTensorFixedPointScratch,
                'fxp4': Int4ActPerTensorFixedPointScratch,
                'fxp2': Int2ActPerTensorFixedPointScratch,
                'ufxp8': Uint8ActPerTensorFixedPointScratch,
                'ufxp4': Uint4ActPerTensorFixedPointScratch,
                'ufxp2': Uint2ActPerTensorFixedPointScratch}

bias_quantizer = {'int8': Int8Bias,
                  'int16': Int16Bias,
                  'int32': Int32Bias}