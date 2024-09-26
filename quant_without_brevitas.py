import torch

from torch import Tensor

import numpy as np



VALUE_ATTR_NAME = 'value'



class IntQuant_np():
    """
    ScriptModule that implements scale, shifted, uniform integer quantization of an input tensor,
    according to an input scale, zero-point and bit-width.

    Args:
        narrow_range (bool): Flag that determines whether restrict quantization to a narrow range or not.
        signed (bool): Flag that determines whether to quantize to a signed range or not.
        float_to_int_impl (Module): Module that performs the conversion from floating point to
            integer representation. Default: RoundSte()
        tensor_clamp_impl (Module): Module that performs clamping. Default: TensorClamp()
        quant_delay_steps (int): Number of training steps to delay quantization for. Default: 0

    Returns:
        Tensor: Quantized output in de-quantized format.

    Examples:
        >>> from brevitas.core.scaling import ConstScaling
        >>> int_quant = IntQuant(narrow_range=True, signed=True)
        >>> scale, zero_point, bit_width = torch.tensor(0.01), torch.tensor(0.), torch.tensor(4.)
        >>> inp = torch.Tensor([0.042, -0.053, 0.31, -0.44])
        >>> out = int_quant(scale, zero_point, bit_width, inp)
        >>> out
        tensor([ 0.0400, -0.0500,  0.0700, -0.0700])

    Note:
        Maps to quant_type == QuantType.INT == 'INT' == 'int' in higher-level APIs.

    Note:
        Set env variable BREVITAS_JIT=1 to enable TorchScript compilation of this module.
    """

    __constants__ = ['signed', 'narrow_range']

    def __init__(
            self,
            narrow_range: bool, 
            signed: bool,):
        super(IntQuant_np, self).__init__()
        self.signed = signed
        self.narrow_range = narrow_range
      

    def tensor_clamp(self,x: np.array, min_val: np.array, max_val: np.array) -> np.array:
        out = torch.where(x > max_val, max_val, x)
        out = np.where(out < min_val, min_val, out)
        return out
    def to_int(self, scale: np.array, zero_point: np.array, bit_width: np.array, x: np.array) -> Tensor:
        y = x / scale
        y = y + zero_point
        min_int_val = self.min_int(bit_width)
        max_int_val = self.max_int(bit_width)
        y = np.round(y)
        y = self.tensor_clamp(y, min_val=min_int_val, max_val=max_int_val)
        return y

    def min_int(self, bit_width):
        return self.min_int(self.signed, self.narrow_range, bit_width)

    
    def max_int(self, bit_width):
        return self.max_int(self.signed, self.narrow_range, bit_width)

   
    def forward(self, scale: np.array, zero_point: np.array, bit_width: np.array, x: np.array) -> Tensor:
        y_int = self.to_int(scale, zero_point, bit_width, x)
        y = y_int - zero_point
        y = y * scale
        
        return y
    def min_int(signed: bool, narrow_range: bool, bit_width: np.array) -> np.array:

        if signed and narrow_range:
            value = -(2 ** (bit_width - 1)) + 1
        elif signed and not narrow_range:
            value = -(2 ** (bit_width - 1))
        else:
            value = 0 * bit_width
        return value
    def max_int(signed: bool, narrow_range: bool, bit_width:  np.array) -> np.array:

        if not signed and not narrow_range:
            value = (2 ** bit_width) - 1
        elif not signed and narrow_range:
            value = (2 ** bit_width) - 2
        else:
            value = (2 ** (bit_width - 1)) - 1
        return value
#----------------------------------------------------------------上面為 IntQuant 的實作，以下為Int8ActPerTensorFloatScratch 的實作
# def _is_signed(quant_injector):
#     if 'signed' in quant_injector:
#         return quant_injector.signed
#     return None


# def _is_narrow_range(quant_injector):
#     if 'narrow_range' in quant_injector:
#         return quant_injector.narrow_range
#     return None
# def _is_groupwise(quant_injector):
#     if 'group_size' in quant_injector:
#         return True
#     else:
#         return False
# def _rounding_mode(quant_injector):
#     if 'float_to_int_impl_type' in quant_injector:
#         return str(quant_injector.float_to_int_impl_type)
#     elif 'float_to_int_impl' in quant_injector:
#         try:
#             impl_type = float_to_int_impl_to_enum(quant_injector.float_to_int_impl)
#             return str(impl_type).upper()
#         except:
#             return None
#     else:
#         return None
# def _update_state_dict_impl(quant_injector):
#     try:
#         impl = quant_injector.update_state_dict_impl
#     except:
#         impl = None
#     return impl

# def register_buffer(self, name: str, tensor: Optional[np.array], persistent: bool = True) -> None:
#     _global_buffer_registration_hooks: Dict[int, Callable] = OrderedDict()
#     _non_persistent_buffers_set: Set[str]
#     super().__setattr__('_non_persistent_buffers_set', set())
#     r"""Add a buffer to the module.

#     This is typically used to register a buffer that should not to be
#     considered a model parameter. For example, BatchNorm's ``running_mean``
#     is not a parameter, but is part of the module's state. Buffers, by
#     default, are persistent and will be saved alongside parameters. This
#     behavior can be changed by setting :attr:`persistent` to ``False``. The
#     only difference between a persistent buffer and a non-persistent buffer
#     is that the latter will not be a part of this module's
#     :attr:`state_dict`.

#     Buffers can be accessed as attributes using given names.

#     Args:
#         name (str): name of the buffer. The buffer can be accessed
#             from this module using the given name
#         tensor (Tensor or None): buffer to be registered. If ``None``, then operations
#             that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
#             the buffer is **not** included in the module's :attr:`state_dict`.
#         persistent (bool): whether the buffer is part of this module's
#             :attr:`state_dict`.

#     Example::

#         >>> # xdoctest: +SKIP("undefined vars")
#         >>> self.register_buffer('running_mean', torch.zeros(num_features))

#     """
#     if persistent is False and isinstance(self, torch.jit.ScriptModule):
#         raise RuntimeError("ScriptModule does not support non-persistent buffers")

#     if '_buffers' not in self.__dict__:
#         raise AttributeError(
#             "cannot assign buffer before Module.__init__() call")
#     elif not isinstance(name, str):
#         raise TypeError(f"buffer name should be a string. Got {torch.typename(name)}")
#     elif '.' in name:
#         raise KeyError("buffer name can't contain \".\"")
#     elif name == '':
#         raise KeyError("buffer name can't be empty string \"\"")
#     elif hasattr(self, name) and name not in self._buffers:
#         raise KeyError(f"attribute '{name}' already exists")
#     elif tensor is not None and not isinstance(tensor, torch.Tensor):
#         raise TypeError(f"cannot assign '{torch.typename(tensor)}' object to buffer '{name}' "
#                         "(torch Tensor or None required)"
#                         )
#     else:
#         for hook in _global_buffer_registration_hooks.values():
#             output = hook(self, name, tensor)
#             if output is not None:
#                 tensor = output
#         self._buffers[name] = tensor
#         if persistent:
#             _non_persistent_buffers_set.discard(name)
#         else:
#             _non_persistent_buffers_set.add(name)


# class SolveActTensorQuantFromEnum():
#     def tensor_quant(quant_type):#目前僅實做到 int8 以及 float32 
#         if quant_type == 'float32':
#             return None
#         elif quant_type == 'int8':
#             return RescalingIntQuant
#         else:
#             raise RuntimeError(f'{quant_type} not recognized.')  
# class QuantProxyProtocol(Protocol):
#     is_quant_enabled: bool
#     is_signed: Optional[bool]
#     is_narrow_range: Optional[bool]
#     rounding_mode: Optional[str]

# class StatelessBuffer():

#     def __init__(self, value: np.array):
#         super(StatelessBuffer, self).__init__() 
#         register_buffer(VALUE_ATTR_NAME, value) 
#     def forward(self):
#         return self.value.detach()


# class QuantProxyFromInjector(QuantProxyProtocol):
#     def __init__(self,  quant_injector: Injector) -> None:
#         QuantProxyProtocol.__init__(self)
#         self.update_state_dict_impl = _update_state_dict_impl(quant_injector)
#         self.quant_injector = quant_injector
#         self.quant_injector = quant_injector.let(proxy_module=self)
#         self._zero_hw_sentinel = StatelessBuffer(np.array(0.0))
#         self.tensor_quant = None
#         # Use a normal list and not a ModuleList since this is a pointer to parent modules
#         self.disable_quant = False

#     @property
#     def requires_export_handler(self):
#         return self.is_quant_enabled
#     def init_tensor_quant(self):
#         self.tensor_quant = self.quant_injector.tensor_quant

#     @property
#     def is_quant_enabled(self):
#         return not self.disable_quant and self.tensor_quant is not None

#     @property
#     def is_signed(self):
#         return _is_signed(self.quant_injector)

#     @property
#     def is_groupwise(self):
#         return _is_groupwise(self.quant_injector)

#     @property
#     def is_narrow_range(self):
#         return _is_narrow_range(self.quant_injector)

#     @property
#     def rounding_mode(self):
#         return _rounding_mode(self.quant_injector)


# class ActQuantProxyFromInjector(QuantProxyFromInjector, ActQuantProxyProtocol):

#     def __init__(self, quant_layer, quant_injector):
#         QuantProxyFromInjector.__init__(self, quant_layer, quant_injector)
#         ActQuantProxyProtocol.__init__(self)
#         self.is_passthrough_act = _is_passthrough_act(quant_injector)

#     @property#定義 is quant_enable
#     def is_quant_enabled(self):
#         return self._is_quant_enabled and not self.disable_quant

#     @is_quant_enabled.setter
#     def is_quant_enabled(self, is_quant_enabled):
#         self._is_quant_enabled = is_quant_enabled


#     def init_tensor_quant(self):#這邊原本有個判斷是否有 acr_impl，但因為這次要實作的 QuantIdentity沒有 act_quant，所以我就把這邊的判斷式刪掉
#         tensor_quant = self.quant_injector.tensor_quant

#         act_impl = None
#         is_act_enabled = False
#         is_quant_enabled = tensor_quant is not None
#         self.is_quant_enabled = is_quant_enabled
#         #因為上面的 is_act_enabled 被設置為 false， 因此將下面註解
#         # if is_act_enabled and is_quant_enabled: 
#         #     self.fused_activation_quant_proxy = FusedActivationQuantProxy(act_impl,tensor_quant)
#         # elif is_act_enabled and not is_quant_enabled:
#         #     self.fused_activation_quant_proxy = FusedActivationQuantProxy(
#         #         act_impl, _TensorQuantDisabledIdentity())
#         if not is_act_enabled and is_quant_enabled:
#             self.fused_activation_quant_proxy = FusedActivationQuantProxy(Identity(), tensor_quant)
#         else:
#             self.fused_activation_quant_proxy = None

#     def scale(self, force_eval=True):
#         current_status = self.training
#         if force_eval:
#             self.eval()
#         scale = self.__call__(self._zero_hw_sentinel()).scale
#         self.train(current_status)
#         return scale

#     def zero_point(self, force_eval=True):
#         current_status = self.training
#         if force_eval:
#             self.eval()
#         zero_point = self.__call__(self._zero_hw_sentinel()).zero_point
#         self.train(current_status)
#         return zero_point

#     def bit_width(self):
#         scale = self.__call__(self._zero_hw_sentinel()).bit_width
#         return scale

#     def forward(self, x: Union[Tensor, np.array]) -> np.array:
#         if self.fused_activation_quant_proxy is not None:
#             y = x
#             if isinstance(y, np.array):
#                 y = y.value
#             # if self.export_mode:
#             #     y = self.fused_activation_quant_proxy.activation_impl(y)
#             #     y = self.export_handler(y)
#             if not self.is_quant_enabled:
#                 y = self.fused_activation_quant_proxy.activation_impl(y)
#             else:
#                 y = self.fused_activation_quant_proxy(y)
#             # If y is an empty QuantTensor, we need to check if this is a passthrough proxy,
#             # otherwise return an empty QuantTensor
#             if isinstance(y, tuple) and not any(map(lambda f: f is None, y)):
#                 return np.array(*y, signed=self.is_signed, training=self.training)
#             elif self.is_passthrough_act:  # preserve scale/zp/bit/sign even without output quant
#                 if isinstance(y, tuple):
#                     y = y[0]
#                 return np.array(y, x.scale, x.zero_point, x.bit_width, x.signed, self.training)
#             else:
#                 if isinstance(y, tuple):
#                     y = y[0]
#                 return np.array(y, training=self.training)
#         else:
#             if isinstance(x, np.array):  # passthrough
#                 return x
#             else:
#                 return np.array(x, training=self.training)
# class ActQuantSolver(SolveActTensorQuantFromEnum):
#      proxy_class = ActQuantProxyFromInjector

# class RescalingIntQuant():
#     """
#     ScriptModule that wraps around an integer quantization implementation like
#     :class:`~brevitas.core.quant.IntQuant`. Scale, zero-point and bit-width are returned from their
#     respective implementations and passed on to the integer quantization implementation.

#     Args:
#         int_quant (Module): Module that implements integer quantization.
#         scaling_impl (Module): Module that takes in the input to quantize and returns a scale factor,
#             here interpreted as threshold on the floating-point range of quantization.
#         int_scaling_impl (Module): Module that takes in a bit-width and returns an integer scale
#             factor, here interpreted as threshold on the integer range of quantization.
#         zero_point_impl (Module): Module that returns an integer zero-point.
#         bit_width_impl (Module): Module that returns a bit-width.

#     Returns:
#         Tuple[Tensor, Tensor, Tensor, Tensor]: Quantized output in de-quantized format, scale,
#             zero-point, bit_width.

#     Examples:
#         >>> from brevitas.core.scaling import ConstScaling
#         >>> from brevitas.core.zero_point import ZeroZeroPoint
#         >>> from brevitas.core.scaling import IntScaling
#         >>> from brevitas.core.quant import IntQuant
#         >>> from brevitas.core.bit_width import BitWidthConst
#         >>> int_quant_wrapper = RescalingIntQuant(
#         ...                         IntQuant(narrow_range=True, signed=True),
#         ...                         ConstScaling(0.1),
#         ...                         IntScaling(signed=True, narrow_range=True),
#         ...                         ZeroZeroPoint(),
#         ...                         BitWidthConst(4))
#         >>> inp = torch.Tensor([0.042, -0.053, 0.31, -0.44])
#         >>> out, scale, zero_point, bit_width = int_quant_wrapper(inp)
#         >>> out
#         tensor([ 0.0429, -0.0571,  0.1000, -0.1000])
#         >>> scale
#         tensor(0.0143)
#         >>> zero_point
#         tensor(0.)
#         >>> bit_width
#         tensor(4.)

#     Note:
#         scale = scaling_impl(x) / int_scaling_impl(bit_width)

#     Note:
#         Set env variable BREVITAS_JIT=1 to enable TorchScript compilation of this module.
#     """

#     def __init__(#sclaing_impl 已經在外面實作過了，所以我現在可以直接導入模組近來
#             self,
#             int_quant: Module,
#             scaling_impl: Module,
#             int_scaling_impl: Module,
#             zero_point_impl = np.array([0]),
#             bit_width_impl= np.array[8]):
#         super(RescalingIntQuant, self).__init__()
#         self.int_quant = int_quant
#         self.scaling_impl = scaling_impl
#         self.int_scaling_impl = int_scaling_impl
#         self.zero_point_impl = zero_point_impl
#         self.msb_clamp_bit_width_impl = bit_width_impl

    
#     def forward(self, x:  np.array) -> Tuple[ np.array,  np.array,  np.array,  np.array]:
#         bit_width = self.msb_clamp_bit_width_impl()
#         threshold = self.scaling_impl(x)
#         int_threshold = self.int_scaling_impl(bit_width)
#         scale = threshold / int_threshold
#         zero_point = self.zero_point_impl(x, scale, bit_width)
#         y = self.int_quant(scale, zero_point, bit_width, x)
#         return y, scale, zero_point, bit_width



# class Int8ActPerTensorFloatScratch(ActQuantSolver):
#     quant_type = 'int8' # integer quantization-----------------------------------------------------------------#SolveActTensorQuantFromEnum
#     bit_width_impl_type = auto() # constant bit width-------------------------------------------------#SolveBitWidthImplFromEnum
#     float_to_int_impl_type = auto() # round to nearest ---------------------------------------------#SolveTensorQuantFloatToIntImplFromEnum 
#     scaling_impl_type = auto() # scale is a parameter initialized from statistics-------#SolveActScalingImplFromEnum
#     scaling_stats_op = auto() # scale statistics is a percentile of the abs value ------------------------#SolveScalingStatsOpFromEnum
#     high_percentile_q = 99.999 # percentile is 99.999
#     collect_stats_steps = 300  # statistics are collected for 300 forward steps before switching to a learned parameter
#     restrict_scaling_type = auto() # scale is a floating-point value------------------------------------#SolveIntScalingImplFromEnum
#     scaling_per_output_channel = False  # scale is per tensor --------------------------------------------------------#SolveActScalingShape
#     bit_width = 8  # bit width is 8
#     signed = True # quantization range is signed
#     narrow_range = False # quantization range is [-128, 127] rather than [-127, 127]
#     zero_point_impl = ZeroZeroPoint # zero point is 0.
