# mengqin@gmail.com || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

import torch
import comfy.ops
import comfy.model_management
import bitsandbytes as bnb
from bitsandbytes.nn.modules import Params4bit
import torch.nn.functional as F
import comfy

class LazyLayer(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_bnb_4bit = False
        self.is_fp8_scaled = False

    def is_bnb_quantized(self):
        return getattr(self, 'is_bnb_4bit', False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        
        weight_key = prefix + 'weight'

        if weight_key in state_dict:

            scale_weight_key = prefix +'scale_weight'
            scale_input_key = prefix + 'scale_input'

            # The dequantization type is determined layer by layer, supporting fp8_scaled and bnb 4bit. 
            # Other types of floating-point numbers are directly supported by the system.
            if state_dict[weight_key].dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and scale_weight_key in state_dict:
                self.is_fp8_scaled = True

                if scale_weight_key in state_dict:                
                    self.register_buffer('scale_weight', state_dict[scale_weight_key])                

                if scale_input_key in state_dict:
                    self.register_buffer('scale_input', state_dict[scale_input_key])
            
            else:
                feature_key = f"{weight_key}.quant_state.bitsandbytes__nf4"
                if feature_key not in state_dict:
                    feature_key = f"{weight_key}.quant_state.bitsandbytes__fp4"

                if feature_key in state_dict:
                    self.is_bnb_4bit = True
                    device = comfy.model_management.get_torch_device()
                    
                    bnb_state_dict = {k: v for k, v in state_dict.items() if k.startswith(weight_key)}
                    weight_data = bnb_state_dict.pop(weight_key)
                    quant_state_dict = {k[len(weight_key)+1:]: v for k, v in bnb_state_dict.items()}

                    bnb_param = Params4bit.from_prequantized(
                        data=weight_data, quantized_stats=quant_state_dict, device=device
                    )
                    self.weight = bnb_param
                    
                    for k in bnb_state_dict.keys():
                        state_dict.pop(k)
                        if k in unexpected_keys: unexpected_keys.remove(k)        
        
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

class LazyOps(comfy.ops.manual_cast):
    class Linear(LazyLayer, comfy.ops.manual_cast.Linear):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.patcher = None            
            self.weight_key_name = None

        def forward(self, x):            
            patcher = getattr(self, "patcher", None)            
            if patcher is not None:
                try:                    
                    _ = patcher
                except ReferenceError:
                    patcher = None

            module_key = getattr(self, "module_key_name", None)
            weight_key = f"{module_key}.weight" if module_key else None
            
            patches_for_this_layer = None
            if patcher is not None and module_key is not None:
                try:
                    if self.is_bnb_quantized():
                        patches_for_this_layer = patcher.bnb_lora_patches.get(module_key, None)
                    else:
                        patches_for_this_layer = patcher.patches.get(weight_key, None)
                except Exception:                    
                    patches_for_this_layer = None

            if getattr(self, "is_bnb_quantized", lambda : False)():                
                if not patches_for_this_layer:
                    bias = self.bias.to(x.dtype) if self.bias is not None else None
                    return bnb.matmul_4bit(
                        x, self.weight.t(), bias=bias, quant_state=getattr(self.weight, "quant_state", None)
                    ).to(x.dtype)

                try:
                    base_w = self.weight.to(x.device)
                    base_dequant = bnb.functional.dequantize_4bit(base_w, base_w.quant_state).to(torch.float32)
                except Exception:
                    base_dequant = self.weight.to(torch.float32).to(x.device)

                weight_final_fp32 = None
                if patcher is not None and module_key:
                    try:
                        weight_final_fp32 = patcher.calculate_weight_with_patches(module_key, base_dequant, is_bnb=True)
                    except Exception:
                        weight_final_fp32 = None
                
                if weight_final_fp32 is None:
                    bias = self.bias.to(x.dtype) if self.bias is not None else None
                    return bnb.matmul_4bit(
                        x, self.weight.t(), bias=bias, quant_state=getattr(self.weight, "quant_state", None)
                    ).to(x.dtype)
                
                weight_final = comfy.float.stochastic_rounding(weight_final_fp32, x.dtype)
                bias = self.bias.to(x.dtype) if self.bias is not None else None
                return F.linear(x, weight_final.to(x.dtype), bias)
            
            elif getattr(self, "is_fp8_scaled", False):
                try:
                    base_weight_dequant = self.weight.to(torch.float32)
                except Exception:
                    base_weight_dequant = self.weight.to(torch.float32)

                scale_weight = getattr(self, 'scale_weight', None)
                if scale_weight is None:
                    scale_weight = torch.tensor(1.0, device=base_weight_dequant.device, dtype=torch.float32)

                try:
                    base_weight_dequant = base_weight_dequant * scale_weight.to(base_weight_dequant.device, torch.float32)
                except Exception:                    
                    try:
                        base_weight_dequant = base_weight_dequant * scale_weight.to(base_weight_dequant.device)
                    except Exception:
                        pass

                weight_final_fp32 = None
                if patcher is not None and module_key:
                    try:
                        weight_final_fp32 = patcher.calculate_weight_with_patches(module_key, base_weight_dequant, is_bnb=False)
                    except Exception:
                        weight_final_fp32 = None

                if weight_final_fp32 is None:
                    weight_final_fp32 = base_weight_dequant

                weight_final = comfy.float.stochastic_rounding(weight_final_fp32, x.dtype)
                bias = self.bias.to(x.dtype) if self.bias is not None else None
                return F.linear(x, weight_final.to(x.dtype), bias)
            
            else:                
                try:
                    return super().forward(x)
                except Exception:                    
                    bias = self.bias.to(x.dtype) if self.bias is not None else None
                    return F.linear(x, self.weight.to(x.dtype), bias)
    
    class Conv2d(comfy.ops.manual_cast.Conv2d): pass
    class Embedding(comfy.ops.manual_cast.Embedding): pass
    class LayerNorm(comfy.ops.manual_cast.LayerNorm): pass
    class GroupNorm(comfy.ops.manual_cast.GroupNorm): pass