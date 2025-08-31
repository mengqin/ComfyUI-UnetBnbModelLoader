# ComfyUI-UnetBnbModelLoader

BitsAndBytes 4bit Quantization (NF4/FP4) Unet Model Loader

ComfyUI has long had issues with support for the bnb 4-bit quantization model. At least the plugins I could find couldn't load most models released on HF. There were also various other issues, such as LoRA incompatibility.

The official NF4 plugin, https://github.com/comfyanonymous/ComfyUI_bitsandbytes_NF4, has been largely abandoned. Previous forks also have varying degrees of issues (model incompatibility or LoRa incompatibility).

The bnb 4-bit format has never been very popular in the community, and GGUF's plugin is very stable and easy to use. So it seems no one has truly developed a plugin with better compatibility.

I used to like GGUF, which offers a variety of model sizes. However, its inference speed is slightly slower, which is a significant issue for models requiring many inference steps.

So, I decided to develop a universal BnB 4-bit model loading plugin, hoping to provide the community with more options.

Features of this plugin:

1. Architecturally agnostic, it supports most popular model plugins(need to be diffusers model or can convert to diffusers model), such as Flux, HiDReam and Qwen-Image, as well as future models.
2. Supports loading sharded safetensors files. This is surprisingly easy to support, and I'm surprised why the standard official model loader doesn't support it, and why other plugins don't support it either.
3. Supports LoRa, with a dedicated dequantization process for LoRa. Perhaps this is the first truly usable BnB 4-bit plugin to support LoRa?
4. 4-bit inference: 4-bit inference is reasonably fast compare to GGUF.
5. On-the-fly dequantization ensures the small size of 4-bit models.
6. Support for independent layer precision allows us to support mixed-precision models.
7. Check if it is a bnb-4bit model when loading, if not fallback to the general unet loader.

## Installation

> [!IMPORTANT]  
> Make sure your ComfyUI is on a recent-enough version to support custom ops when loading the UNET-only.

To install the custom node normally, git clone this repository into your custom nodes folder (`ComfyUI/custom_nodes`) and install the only dependency for inference (`pip install --upgrade bitsandbytes`)

```
git clone https://github.com/mengqin/ComfyUI-UnetBnbModelLoader
```

To install the custom node on a standalone ComfyUI release, open a CMD inside the "ComfyUI_windows_portable" folder (where your `run_nvidia_gpu.bat` file is) and use the following commands:

```
git clone https://github.com/mengqin/ComfyUI-UnetBnbModelLoader ComfyUI/custom_nodes/ComfyUI-UnetBnbModelLoader
.\python_embeded\python.exe -s -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-UnetBnbModelLoader\requirements.txt
```

Because this plugin relies on bitsandbytes, we are unable to support macOS and AMD GPUs.

## Usage

After installation, double-click a blank space in comfyui, search for "Unet Bnb Model Loader," and select your model to use it. Please place your model in the unet or diffuser-models directory.
If your model is multi-sharded, remember to place all shards in the same directory and maintain the classic 0000n-of-0000m shard model name format. In our model drop-down list, multi-shard models will not display the specific model name, but the model directory instead.

Supported model files:

- [flux1-dev-bnb-nf4](https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4)
- [Flux-Krea_bnb-nf4](https://huggingface.co/AcademiaSD/Flux-Krea_bnb-nf4)
- [flux1-schnell-nf4-v2](https://huggingface.co/duuuuuuuden/flux1-schnell-nf4-v2)
- [flux1-nf4-unet](https://huggingface.co/silveroxides/flux1-nf4-unet)
- [HiDream-I1-Fast-nf4](https://huggingface.co/azaneko/HiDream-I1-Fast-nf4)
- [HiDream-I1-Full-nf4](https://huggingface.co/azaneko/HiDream-I1-Full-nf4)
- [HiDream-I1-Dev-nf4](https://huggingface.co/azaneko/HiDream-I1-Dev-nf4)
- [qwen-image-4bit](https://huggingface.co/ovedrive/qwen-image-4bit)

Some models are converted without first being converted to diffusers models. Instead, BNB 4-bit quantization is performed directly on the original model. This will cause comfyui to fail to correctly perform mmdit conversion before loading these models. This is because they cannot recognize and correctly handle the newly added quantized vector format of BNB 4-bit.

Unsupported models:

- [flux.1-schnell-nf4](https://huggingface.co/gradjitta/flux.1-schnell-nf4)
- [flux1-schnell-bnb-nf4](https://huggingface.co/Keffisor21/flux1-schnell-bnb-nf4)
- [sd35-large-nf4](https://huggingface.co/sayakpaul/sd35-large-nf4)
