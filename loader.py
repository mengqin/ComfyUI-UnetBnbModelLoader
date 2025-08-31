# mengqin@gmail.com || Apache-2.0 (apache.org/licenses/LICENSE-2.0)

import logging
import os
import re
import safetensors
import torch
from comfy.cli_args import args

MMAP_TORCH_FILES = args.mmap_torch_files
DISABLE_MMAP = args.disable_mmap

ALWAYS_SAFE_LOAD = False
if hasattr(torch.serialization, "add_safe_globals"):  # TODO: this was added in pytorch 2.4, the unsafe path should be removed once earlier versions are deprecated
    ALWAYS_SAFE_LOAD = True

# Rewritten from load_troch_file() in utils.py to increase the ability to read shard files.
def safetensors_sd_loader(ckpt, safe_load=False, device=None, return_metadata=False):
    if device is None:
        device = torch.device("cpu")
    metadata = None
    
    def _load_single_safetensor_file(path, sd, metadata_holder):
        try:
            with safetensors.safe_open(path, framework="pt", device=device.type) as f:
                for k in f.keys():
                    tensor = f.get_tensor(k)
                    if DISABLE_MMAP:  # TODO: Not sure if this is the best way to bypass the mmap issues
                        tensor = tensor.to(device=device, copy=True)
                    if k in sd:                        
                        raise ValueError(f"Duplicate tensor key '{k}' found while loading shard {path}.")
                    sd[k] = tensor
                if return_metadata:
                    m = f.metadata()                    
                    if metadata_holder[0] is None and m:
                        metadata_holder[0] = m
                    elif m and metadata_holder[0] is not None and m != metadata_holder[0]:
                        logging.warning("safetensors shard metadata mismatch: file %s metadata differs from previous shards.", path)
        except Exception as e:
            if len(e.args) > 0:
                message = e.args[0]
                if "HeaderTooLarge" in message:
                    raise ValueError("{}\n\nFile path: {}\n\nThe safetensors file is corrupt or invalid. Make sure this is actually a safetensors file and not a ckpt or pt or other filetype.".format(message, path))
                if "MetadataIncompleteBuffer" in message:
                    raise ValueError("{}\n\nFile path: {}\n\nThe safetensors file is corrupt/incomplete. Check the file size and make sure you have copied/downloaded it correctly.".format(message, path))
            raise

    # If ckpt is a path to a directory try to find shards inside
    try:
        if isinstance(ckpt, str) and os.path.isdir(ckpt):
            dirpath = ckpt
            files = os.listdir(dirpath)
            # pattern to detect HF-style shards: prefix<idx>-of-<total>.safetensors or .sft
            shard_re = re.compile(r"^(.+)-(\d+)-of-(\d+)\.(safetensors|sft)$", flags=re.IGNORECASE)
            groups = {}
            for fname in files:
                m = shard_re.match(fname)
                if m:
                    prefix = m.group(1)
                    idx = int(m.group(2))                    
                    groups.setdefault(prefix, []).append((idx, fname))
            selected_files = []
            if groups:                
                best_prefix, members = max(groups.items(), key=lambda kv: len(kv[1]))
                members_sorted = sorted(members, key=lambda t: t[0])
                selected_files = [os.path.join(dirpath, fn) for (_, fn) in members_sorted]
            else:
                # fallback: load all .safetensors/.sft in the directory (if any)
                safetensor_files = sorted([
                    os.path.join(dirpath, f) for f in files
                    if f.lower().endswith(".safetensors") or f.lower().endswith(".sft")
                ])
                if safetensor_files:
                    selected_files = safetensor_files

            if selected_files:
                sd = {}
                metadata_holder = [None]
                for path in selected_files:
                    _load_single_safetensor_file(path, sd, metadata_holder)
                metadata = metadata_holder[0]
                return (sd, metadata) if return_metadata else sd            
    except Exception:        
        raise

    # If ckpt is a path to a shard file (one shard provided), attempt to find sibling shards
    if isinstance(ckpt, str) and (ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft")) and os.path.isfile(ckpt):
        # try to detect HF-style shard naming on this filename
        shard_re = re.compile(r"^(.+)-(\d+)-of-(\d+)\.(safetensors|sft)$", flags=re.IGNORECASE)
        base_name = os.path.basename(ckpt)
        m = shard_re.match(base_name)
        if m:
            prefix = m.group(1)
            dirpath = os.path.dirname(ckpt) or "."            
            sibling_pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)-of-(\d+)\.(safetensors|sft)$", flags=re.IGNORECASE)
            siblings = []
            for fname in os.listdir(dirpath):
                sm = sibling_pattern.match(fname)
                if sm:
                    idx = int(sm.group(1))
                    siblings.append((idx, os.path.join(dirpath, fname)))
            if siblings:
                siblings_sorted = [p for (_, p) in sorted(siblings, key=lambda t: t[0])]
                sd = {}
                metadata_holder = [None]
                for path in siblings_sorted:
                    _load_single_safetensor_file(path, sd, metadata_holder)
                metadata = metadata_holder[0]
                return (sd, metadata) if return_metadata else sd            

    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):        
        try:
            sd = {}
            metadata_holder = [None]
            _load_single_safetensor_file(ckpt, sd, metadata_holder)
            metadata = metadata_holder[0]
            return (sd, metadata) if return_metadata else sd
        except Exception:
            raise
    
    torch_args = {}
    if MMAP_TORCH_FILES:
        torch_args["mmap"] = True

    if safe_load or ALWAYS_SAFE_LOAD:
        pl_sd = torch.load(ckpt, map_location=device, weights_only=True, **torch_args)
    else:
        logging.warning("WARNING: loading {} unsafely, upgrade your pytorch to 2.4 or newer to load this file safely.".format(ckpt))
        pl_sd = torch.load(ckpt, map_location=device, pickle_module=comfy.checkpoint_pickle)
    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        if len(pl_sd) == 1:
            key = list(pl_sd.keys())[0]
            sd = pl_sd[key]
            if not isinstance(sd, dict):
                sd = pl_sd
        else:
            sd = pl_sd
    return (sd, metadata) if return_metadata else sd