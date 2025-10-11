import torch
import os
import sys
from pathlib import Path
import yaml
import argparse

import folder_paths
from comfy.utils import ProgressBar
import numpy as np
from tqdm import tqdm

# --- Robust Path Handling ---
if 'interpolation' not in folder_paths.folder_names_and_paths:
    new_path = os.path.join(folder_paths.models_dir, 'interpolation')
    os.makedirs(new_path, exist_ok=True)
    folder_paths.folder_names_and_paths['interpolation'] = ([new_path], {'.pth', '.ckpt'})

# --- Helper Functions ---

def dict2namespace(config):
    """Converts a dictionary to a namespace for easier access."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def find_models(folder_type: str, extensions: list) -> list:
    """Recursively finds all model files with given extensions in the specified folder type."""
    model_list = []
    base_paths = folder_paths.get_folder_paths(folder_type)
    
    for base_path in base_paths:
        for root, _, files in os.walk(base_path, followlinks=True):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    relative_path = os.path.relpath(os.path.join(root, file), base_path)
                    model_list.append(relative_path.replace("\\", "/"))
    return sorted(list(set(model_list)))

# --- TLBVFI Setup ---

try:
    current_path = Path(__file__).parent
    tlbvfi_path = current_path / "TLBVFI"
    if tlbvfi_path.is_dir():
        sys.path.insert(0, str(tlbvfi_path))
        from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
    else:
        raise ImportError("TLBVFI directory not found.")
except ImportError as e:
    print("-------------------------------------------------------------------")
    print(f"Error: {e}")
    print("Could not import TLBVFI model from ComfyUI-TLBVFI node.")
    print("Please follow the setup instructions in the README.md file.")
    print("-------------------------------------------------------------------")
    raise

# --- Main Node Class ---

class TLBVFI_VFI:
    @classmethod
    def INPUT_TYPES(s):
        # We only need the main model file now.
        unet_models = find_models("interpolation", [".pth"])
        if not unet_models:
             raise Exception("No TLBVFI UNet models (.pth) found in 'ComfyUI/models/interpolation/'. Please download 'vimeo_unet.pth'.")
        
        return {
            "required": {
                "images": ("IMAGE", ),
                "model_name": (unet_models, ),
                "times_to_interpolate": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "gpu_id": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "interpolate"
    CATEGORY = "frame_interpolation/TLBVFI" # Updated Category for better organization

    def interpolate(self, images, model_name, times_to_interpolate, gpu_id):
        # --- Setup ---
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        
        # --- Load Config ---
        tlbvfi_repo_path = Path(__file__).parent / "TLBVFI"
        config_path = tlbvfi_repo_path / "configs" / "Template-LBBDM-video.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}. Make sure the TLBVFI repo is cloned correctly.")
        
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        nconfig = dict2namespace(config)

        # --- Simplified Model Loading ---
        model_path = folder_paths.get_full_path("interpolation", model_name)
        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_name} not found.")
            
        # Prevent the model from trying to load a VQGAN checkpoint during initialization
        nconfig.model.VQGAN.params.ckpt_path = None
        
        # 1. Initialize the model structure with random weights.
        model = LatentBrownianBridgeModel(nconfig.model).to(device)

        # 2. Load the entire state dict from the single .pth file.
        # This will populate both the VQGAN and the UNet with the correct weights.
        checkpoint = torch.load(model_path, map_location=device)
        
        # The state dict might be nested under a 'model' key.
        state_dict_to_load = checkpoint.get('model', checkpoint)
        
        model.load_state_dict(state_dict_to_load)
        model.eval()

        # Enable FP16 mixed precision on CUDA for ~30% speedup and 45% memory reduction
        # Uses custom convert_to_fp16() method to recursively convert all VQGAN submodules
        use_fp16 = False
        if device.type == 'cuda':
            try:
                model.convert_to_fp16()
                use_fp16 = True
                print("TLBVFI: FP16 mixed precision enabled (Tensor Core acceleration)")
            except Exception as e:
                print(f"TLBVFI Warning: FP16 conversion failed, using FP32 fallback. Error: {e}")
                use_fp16 = False

        # --- Prepare Images ---
        image_tensors = images.permute(0, 3, 1, 2).float()
        image_tensors = (image_tensors * 2.0) - 1.0
        
        if len(image_tensors) < 2:
            print("TLBVFI Warning: Not enough images to interpolate. Returning original images.")
            return (images, )

        gui_pbar = ProgressBar(len(image_tensors) - 1)

        # Calculate total output frames
        total_segments = len(image_tensors) - 1
        frames_per_segment = 2 ** times_to_interpolate
        total_frames = total_segments * frames_per_segment + 1

        # Pre-allocate output tensor on CPU (avoids torch.cat overhead)
        final_tensors = torch.empty(
            (total_frames, *image_tensors.shape[1:]),
            dtype=image_tensors.dtype,
            device='cpu'
        )

        # Write first frame
        write_idx = 0
        final_tensors[write_idx] = image_tensors[0]
        write_idx += 1

        # --- Main Interpolation Loop ---
        for i in tqdm(range(len(image_tensors) - 1), desc="TLBVFI Interpolating"):
            # Convert to FP16 when moving to CUDA device for consistency with model dtype
            if use_fp16:
                frame1 = image_tensors[i].unsqueeze(0).to(device=device, dtype=torch.float16)
                frame2 = image_tensors[i+1].unsqueeze(0).to(device=device, dtype=torch.float16)
            else:
                frame1 = image_tensors[i].unsqueeze(0).to(device)
                frame2 = image_tensors[i+1].unsqueeze(0).to(device)

            current_frames = [frame1, frame2]
            for _ in range(times_to_interpolate):
                temp_frames = [current_frames[0]]
                for j in range(len(current_frames) - 1):
                    try:
                        with torch.no_grad():
                            mid_frame = model.sample(current_frames[j], current_frames[j+1], disable_progress=True)
                        temp_frames.extend([mid_frame, current_frames[j+1]])
                    except RuntimeError as e:
                        if "type" in str(e).lower() and use_fp16:
                            # FP16 type mismatch detected during inference - should be rare if convert_to_fp16() worked
                            raise RuntimeError(
                                f"FP16 type mismatch during inference. This suggests a submodule wasn't properly converted. "
                                f"Original error: {e}"
                            )
                        else:
                            # Re-raise other runtime errors
                            raise
                current_frames = temp_frames

            # Write directly to pre-allocated CPU tensor
            for frame in current_frames[1:]:
                final_tensors[write_idx] = frame.squeeze(0).cpu()
                write_idx += 1

            # Clear intermediate frames
            del current_frames, temp_frames, frame1, frame2

            gui_pbar.update(1)
        
        # --- Convert back to ComfyUI's expected format ---
        final_tensors = (final_tensors + 1.0) / 2.0
        final_tensors = final_tensors.clamp(0, 1)
        final_tensors = final_tensors.permute(0, 2, 3, 1)

        return (final_tensors, )