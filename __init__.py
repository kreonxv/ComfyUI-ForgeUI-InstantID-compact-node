import requests
import json
import base64
import io
import numpy as np
from PIL import Image
import torch

def get_forge_models(forge_url="http://127.0.0.1:7860"):
    """Query Forge UI for available ControlNet models"""
    try:
        response = requests.get(f"{forge_url}/controlnet/model_list", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if "model_list" in data:
            models = data["model_list"]
            
            # Categorize models
            instantid_ip = [m for m in models if "instant" in m.lower() and "ip-adapter" in m.lower()]
            instantid_keypoints = [m for m in models if "instant" in m.lower() and "control" in m.lower()]
            canny = [m for m in models if "canny" in m.lower()]
            
            return {
                "instantid_ip": instantid_ip if instantid_ip else ["ip-adapter_instant_id_sdxl"],
                "instantid_keypoints": instantid_keypoints if instantid_keypoints else ["control_instant_id_sdxl"],
                "canny": canny if canny else ["diffusers_xl_canny_full"]
            }
    except:
        pass
    
    # Return defaults if connection fails
    return {
        "instantid_ip": ["ip-adapter_instant_id_sdxl"],
        "instantid_keypoints": ["control_instant_id_sdxl"],
        "canny": ["diffusers_xl_canny_full"]
    }

def get_forge_loras(forge_url="http://127.0.0.1:7860"):
    """Query Forge UI for available LoRA models"""
    try:
        response = requests.get(f"{forge_url}/sdapi/v1/loras", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0:
            # Extract just the names without file extensions
            lora_names = [item.get("name", "").replace(".safetensors", "").replace(".pt", "").replace(".ckpt", "") 
                         for item in data if "name" in item]
            lora_names = [name for name in lora_names if name]  # Remove empty strings
            
            if lora_names:
                # Add "None" option at the beginning
                return ["None"] + lora_names
    except:
        pass
    
    # Return defaults if connection fails
    return ["None", "dmd2_sdxl_4step_lora_fp16"]

def get_forge_sd_models(forge_url="http://127.0.0.1:7860"):
    """Query Forge UI for available SD checkpoint models"""
    try:
        response = requests.get(f"{forge_url}/sdapi/v1/sd-models", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0:
            # Extract model titles
            model_titles = [item.get("title", item.get("model_name", "")) for item in data]
            model_titles = [title for title in model_titles if title]  # Remove empty strings
            
            if model_titles:
                return model_titles
    except:
        pass
    
    # Return defaults if connection fails
    return ["zep7_v2.safetensors"]

class ForgeUIController:
    """
    ComfyUI node to control a LOCAL Forge UI / Automatic1111 WebUI API with InstantID + Canny ControlNet workflow.
    NOTE: This requires a locally running Forge UI instance. For cloud-based generation, use the ForgeAPI nodes instead.
    """
    
    def __init__(self):
        self.forge_url = "http://127.0.0.1:7860"
    
    @classmethod
    def INPUT_TYPES(cls):
        # Try to get models and loras from Forge UI
        forge_models = get_forge_models()
        forge_loras = get_forge_loras()
        forge_sd_models = get_forge_sd_models()
        
        return {
            "required": {
                "forge_url": ("STRING", {"default": "http://127.0.0.1:7860", "multiline": False}),
                "sd_model": (forge_sd_models,),
                "instantid_model": (forge_models["instantid_ip"],),
                "instantid_keypoints_model": (forge_models["instantid_keypoints"],),
                "canny_model": (forge_models["canny"],),
                "lora_model": (forge_loras, {"default": "dmd2_sdxl_4step_lora_fp16"}),
                "lora_weight": ("FLOAT", {"default": 1.1, "min": 0.0, "max": 2.0, "step": 0.05}),
                "prompt": ("STRING", {"multiline": True, "default": "a well dressed woman"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "anime, 3d, painting, cartoon"}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 150}),
                "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "width": ("INT", {"default": 1152, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1920, "min": 64, "max": 2048, "step": 8}),
                "seed": ("INT", {"default": 42, "min": -1, "max": 0xffffffffffffffff}),
                "sampler_name": (["LCM", "Euler", "Euler a", "DPM++ 2M", "DPM++ SDE", "DDIM"], {"default": "LCM"}),
                "scheduler": (["Exponential", "Karras", "Normal", "Simple", "SGM Uniform"], {"default": "Exponential"}),
            },
            "optional": {
                "instantid_image": ("IMAGE",),
                "canny_image": ("IMAGE",),
                "instantid_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "instantid_keypoints_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "canny_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "canny_start_step": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "canny_end_step": ("FLOAT", {"default": 0.51, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "ForgeUI"
    
    def tensor_to_base64(self, image_tensor):
        """Convert ComfyUI image tensor to base64 string"""
        # ComfyUI images are in format [B, H, W, C] with values 0-1
        i = 255. * image_tensor[0].cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return img_base64
    
    def base64_to_tensor(self, base64_str):
        """Convert base64 string to ComfyUI image tensor"""
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Convert to ComfyUI format [1, H, W, C]
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        
        return torch.from_numpy(img_array)[None,]
    
    def generate(self, forge_url, sd_model, instantid_model, instantid_keypoints_model, canny_model,
                 lora_model, lora_weight, prompt, negative_prompt, steps, cfg_scale, width, height, seed,
                 sampler_name, scheduler, instantid_image=None, canny_image=None,
                 instantid_weight=1.0, instantid_keypoints_weight=1.0, canny_weight=1.0, 
                 canny_start_step=0.0, canny_end_step=0.51):
        
        # Use provided URL
        self.forge_url = forge_url.rstrip('/')
        
        # Check if URL is empty or disabled
        if not self.forge_url or self.forge_url.lower() in ['disabled', 'none', 'skip']:
            print("ForgeUIController: URL is disabled, skipping local Forge UI request")
            # Return a placeholder black image
            placeholder = torch.zeros((1, height, width, 3))
            return (placeholder,)
        
        # Construct LoRA tag from dropdown and weight
        lora_tag = ""
        if lora_model and lora_model.lower() != "none":
            lora_tag = f"<lora:{lora_model}:{lora_weight}>"
        
        # Add LoRA tag to prompt
        full_prompt = f"{prompt} {lora_tag}".strip()
        
        # Build the payload
        payload = {
            "prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "distilled_cfg_scale": 3.5,
            "width": width,
            "height": height,
            "seed": seed,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "batch_size": 1,
            "n_iter": 1,
            "override_settings": {
                "sd_model_checkpoint": sd_model
            },
            "alwayson_scripts": {
                "ControlNet": {
                    "args": []
                }
            }
        }
        
        # ControlNet 0: InstantID IP-Adapter
        controlnet_0 = {
            "enabled": True,
            "module": "InsightFace (InstantID)",
            "model": instantid_model,
            "weight": instantid_weight,
            "resize_mode": "Crop and Resize",
            "guidance_start": 0.0,
            "guidance_end": 1.0,
            "pixel_perfect": False,
            "control_mode": "Balanced",
            "processor_res": 512,
            "threshold_a": 0.5,
            "threshold_b": 0.5,
        }
        
        if instantid_image is not None:
            controlnet_0["image"] = self.tensor_to_base64(instantid_image)
        
        # ControlNet 1: InstantID Face Keypoints
        controlnet_1 = {
            "enabled": True,
            "module": "instant_id_face_keypoints",
            "model": instantid_keypoints_model,
            "weight": instantid_keypoints_weight,
            "resize_mode": "Crop and Resize",
            "guidance_start": 0.0,
            "guidance_end": 1.0,
            "pixel_perfect": False,
            "control_mode": "ControlNet is more important",
            "processor_res": 512,
            "threshold_a": 0.5,
            "threshold_b": 0.5,
        }
        
        if instantid_image is not None:
            controlnet_1["image"] = self.tensor_to_base64(instantid_image)
        
        # ControlNet 2: Canny (also uses canny_image for control_instant_id_sdxl keypoints)
        controlnet_2 = {
            "enabled": True,
            "module": "canny",
            "model": canny_model,
            "weight": canny_weight,
            "resize_mode": "Crop and Resize",
            "guidance_start": canny_start_step,
            "guidance_end": canny_end_step,
            "pixel_perfect": False,
            "control_mode": "ControlNet is more important",
            "processor_res": 512,
            "threshold_a": 100,
            "threshold_b": 200,
        }
        
        if canny_image is not None:
            controlnet_2["image"] = self.tensor_to_base64(canny_image)
            # Also apply canny image to ControlNet 1 (instant_id_face_keypoints)
            controlnet_1["image"] = self.tensor_to_base64(canny_image)
        
        # Add all ControlNets to payload
        payload["alwayson_scripts"]["ControlNet"]["args"] = [
            controlnet_0,
            controlnet_1,
            controlnet_2
        ]
        
        # Make the API request
        try:
            print(f"Sending request to Forge UI at {self.forge_url}/sdapi/v1/txt2img")
            response = requests.post(
                f"{self.forge_url}/sdapi/v1/txt2img",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the first image from the response
            if "images" in result and len(result["images"]) > 0:
                image_b64 = result["images"][0]
                output_image = self.base64_to_tensor(image_b64)
                print("Image generated successfully!")
                return (output_image,)
            else:
                raise Exception("No images returned from Forge UI")
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Forge UI: {e}")
            raise Exception(f"Failed to connect to Forge UI at {self.forge_url}: {e}")
        except Exception as e:
            print(f"Error generating image: {e}")
            raise


# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "ForgeUIController": ForgeUIController
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ForgeUIController": "Local Forge UI Controller (InstantID + Canny)"
}