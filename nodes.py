# mengqin@gmail.com || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

import logging
import os
import re
import uuid
import copy
import weakref
import torch

import comfy.sd
import comfy.model_patcher
import folder_paths

from .ops import LazyOps
from .loader import safetensors_sd_loader

class UnetBnbModelPatcher(comfy.model_patcher.ModelPatcher):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # module_key list of (strength_patch, patch_obj, strength_model, None, None)
        self.bnb_lora_patches = {}
        # optional backups for module-local temp data when partially_unload
        self._bnb_lora_module_backups = {}

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):        
        added = []
        for key in patches:
            if not isinstance(key, str):
                continue

            module_key = key.rsplit('.', 1)[0]
            try:
                module = comfy.utils.get_attr(self.model, module_key)
                is_bnb = hasattr(module, 'is_bnb_quantized') and module.is_bnb_quantized()
            except Exception:                
                is_bnb = False

            if is_bnb:
                self.bnb_lora_patches.setdefault(module_key, []).append(
                    (strength_patch, patches[key], strength_model, None, None)
                )
            else:
                current = self.patches.get(key, [])
                current.append((strength_patch, patches[key], strength_model, None, None))
                self.patches[key] = current
            added.append(key)

        self.patches_uuid = uuid.uuid4()
        return added

    def clone(self):
        cloned = super().clone()
        if not isinstance(cloned, UnetBnbModelPatcher):
            new_cloned = UnetBnbModelPatcher(cloned.model, cloned.load_device, cloned.offload_device, cloned.size)
            new_cloned.patches = cloned.patches
            new_cloned.object_patches = cloned.object_patches
            cloned = new_cloned
        
        cloned.bnb_lora_patches = copy.deepcopy(self.bnb_lora_patches)
        cloned._bnb_lora_module_backups = {}
        return cloned

    def pre_run(self, *args, **kwargs):        
        super().pre_run(*args, **kwargs)

        for name, module in self.model.named_modules():            
            try:
                if isinstance(module, LazyOps.Linear):                    
                    try:
                        module.patcher = weakref.proxy(self)
                    except Exception:                        
                        module.patcher = self
                    
                    module.module_key_name = name
                    module.weight_key_name = f"{name}.weight"
            except Exception:                
                logging.debug(f"pre_run: skip module {name} assignment due to exception", exc_info=True)
        
        self.apply_bnb_patches()

    def apply_bnb_patches(self):        
        for module_key, p_list in list(self.bnb_lora_patches.items()):
            try:
                module = comfy.utils.get_attr(self.model, module_key)
            except Exception:
                continue
            if getattr(module, "_bnb_lora_attached", False):
                continue            
            module._bnb_lora_attached = True
            module._bnb_lora_patch_count = len(p_list)

    def get_patches_for_module(self, module_key_name, *, is_bnb=False):        
        if is_bnb:
            return self.bnb_lora_patches.get(module_key_name, None)
        else:
            return self.patches.get(f"{module_key_name}.weight", None)

    def remove_bnb_patches(self):        
        for module_key in list(self.bnb_lora_patches.keys()):
            try:
                module = comfy.utils.get_attr(self.model, module_key)
            except Exception:
                continue

            for attr in ("_bnb_lora_attached", "_bnb_lora_patch_count",):
                if hasattr(module, attr):
                    try:
                        delattr(module, attr)
                    except Exception:
                        logging.debug(f"remove_bnb_patches: could not del {attr} on {module_key}", exc_info=True)
            
            self._bnb_lora_module_backups.pop(module_key, None)

    def clear_bnb_patches(self):        
        self.bnb_lora_patches.clear()
        self._bnb_lora_module_backups.clear()

    def unpatch_model(self, device_to=None, unpatch_weights=True):        
        super().unpatch_model(device_to=device_to, unpatch_weights=unpatch_weights)

        self.remove_bnb_patches()
        
        for name, module in self.model.named_modules():            
            if hasattr(module, "patcher"):
                try:
                    p = getattr(module, "patcher")                    
                    delattr(module, "patcher")
                except Exception:
                    pass

            for attr in ("module_key_name", "weight_key_name"):
                if hasattr(module, attr):
                    try:
                        delattr(module, attr)
                    except Exception:
                        pass
        
        return

    def partially_unload(self, device_to, memory_to_free=0):

        memory_freed = super().partially_unload(device_to, memory_to_free=memory_to_free)

        for name, module in self.model.named_modules():
            if getattr(module, "_bnb_lora_attached", False):
                module_key = getattr(module, "module_key_name", name)                
                if hasattr(module, "_bnb_lora_patch_count"):
                    self._bnb_lora_module_backups[module_key] = getattr(module, "_bnb_lora_patch_count", None)                
                for attr in ("_bnb_lora_attached", "_bnb_lora_patch_count",):
                    if hasattr(module, attr):
                        try:
                            delattr(module, attr)
                        except Exception:
                            pass
        return memory_freed
    
    def calculate_weight_with_patches(self, module_key_name, base_weight_fp32, is_bnb=False):                
        if is_bnb:
            patches = self.bnb_lora_patches.get(module_key_name, None)
        else:
            patches = self.patches.get(f"{module_key_name}.weight", None)

        if not patches:
            return None
        
        try:            
            base = base_weight_fp32.to(torch.float32)
            weight_final_fp32 = comfy.lora.calculate_weight(patches, base, f"{module_key_name}.weight")            
            return weight_final_fp32.to(torch.float32)
        except Exception:            
            logging.exception(f"calculate_weight_with_patches failed for {module_key_name}")
            return None
        
def get_safetensors_model_list(folder_path_key):
    shard_pattern = re.compile(r'.*-(\d{5})-of-(\d{5})\.safetensors$')
    try:
        initial_list = folder_paths.get_filename_list(folder_path_key)
    except KeyError:
        logging.error(f"Path type '{folder_path_key}' is not registered.")
        return []

    sharded_files_to_remove = set()
    parent_dirs_to_add = set()

    for item in initial_list:        
        is_dir = False
        for basedir in folder_paths.get_folder_paths(folder_path_key):
             if os.path.isdir(os.path.join(basedir, item)):
                 is_dir = True
                 break
        if is_dir:
            continue

        filename = os.path.basename(item)
        if shard_pattern.match(filename):
            sharded_files_to_remove.add(item)
            parent_dir = os.path.dirname(item)
            if parent_dir and parent_dir != ".":                
                parent_dirs_to_add.add(parent_dir.replace(os.sep, '/'))

    final_list = [item for item in initial_list if item not in sharded_files_to_remove]
    final_list.extend(list(parent_dirs_to_add))
    
    return sorted(list(set(final_list)))

def is_bnb_4bit(sd: dict) -> bool:
    if not isinstance(sd, dict):
        return False
    for k in sd.keys():
        if k.endswith(".quant_state.bitsandbytes__nf4") or k.endswith(".quant_state.bitsandbytes__fp4"):
            return True
    return False

class UnetBnbModelLoader:
    FOLDER_PATH_KEY = "unet"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_safetensors_model_list(s.FOLDER_PATH_KEY),),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "loaders"
    TITLE = "Unet Bnb Model Loader"

    def load_model(self, model_name):        
        model_path = folder_paths.get_full_path(self.FOLDER_PATH_KEY, model_name)
        
        if model_path is None:
            for basedir in folder_paths.get_folder_paths(self.FOLDER_PATH_KEY):
                candidate_path = os.path.join(basedir, model_name)
                if os.path.isdir(candidate_path):
                    model_path = candidate_path
                    break
        
        if model_path is None:
            raise FileNotFoundError(f"Model not found in the directory configured by '{self.FOLDER_PATH_KEY}' class: {model_name}")        
        
        state_dict = safetensors_sd_loader(model_path)        
        
        if is_bnb_4bit(state_dict):
            model_patcher  = comfy.sd.load_diffusion_model_state_dict(state_dict, {"custom_operations": LazyOps()})
            custom_patcher = UnetBnbModelPatcher(model_patcher.model, model_patcher.load_device, model_patcher.offload_device, model_patcher.size)
        else:
            model_patcher  = comfy.sd.load_diffusion_model_state_dict(state_dict)
            custom_patcher = comfy.model_patcher.ModelPatcher(model_patcher.model, model_patcher.load_device, model_patcher.offload_device, model_patcher.size)
        
        if model_patcher is None:
            raise RuntimeError(f"Unable to detect or load UNet model: {model_path}")        
        
        return (custom_patcher,)

NODE_CLASS_MAPPINGS = {
    "UnetBnbModelLoader": UnetBnbModelLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "UnetBnbModelLoader": "Unet Bnb Model Loader",
}