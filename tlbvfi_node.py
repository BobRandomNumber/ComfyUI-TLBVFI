import torch
import os
import sys
from pathlib import Path
import yaml
import folder_paths
from comfy_api.latest import io, ComfyExtension

# Setup models directory for frame interpolation
if 'interpolation' not in folder_paths.folder_names_and_paths:
    new_path = os.path.join(folder_paths.models_dir, 'interpolation')
    os.makedirs(new_path, exist_ok=True)
    folder_paths.folder_names_and_paths['interpolation'] = ([new_path], {'.pth', '.ckpt'})

_CURRENT_MODEL = None
_CURRENT_MODEL_KEY = None

def find_models(folder_type: str, extensions: list) -> list:
    model_list = []
    base_paths = folder_paths.get_folder_paths(folder_type)
    for base_path in base_paths:
        for root, _, files in os.walk(base_path, followlinks=True):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    relative_path = os.path.relpath(os.path.join(root, file), base_path)
                    model_list.append(relative_path.replace("\\", "/"))
    return sorted(list(set(model_list)))

class TLBVFI_VFI(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        unet_models = find_models("interpolation", [".pth"])
        return io.Schema(
            node_id="TLBVFI_VFI",
            display_name="TLBVFI Frame Interpolation",
            category="frame_interpolation/TLBVFI",
            description="Temporal-Aware Latent Brownian Bridge for Video Frame Interpolation",
            inputs=[
                io.Image.Input("images"),
                io.Combo.Input("model_name", options=unet_models if unet_models else ["No models found"]),
                io.Int.Input("times_to_interpolate", default=1, min=1, max=4, step=1),
                io.Int.Input("diffusion_steps", default=10, min=1, max=100, step=1),
                io.Int.Input("batch_size", default=2, min=1, max=64),
                io.Float.Input("flow_scale", default=0.5, min=0.1, max=1.0, step=0.1),
            ],
            outputs=[io.Image.Output()]
        )

    @classmethod
    def execute(cls, images, model_name, times_to_interpolate, diffusion_steps, batch_size, flow_scale) -> io.NodeOutput:
        from comfy.utils import ProgressBar
        from tqdm import tqdm
        import gc

        if model_name == "No models found":
             raise Exception("No TLBVFI UNet models found. Please download 'vimeo_unet.pth' to models/interpolation.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        current_path = Path(__file__).parent
        tlbvfi_path = current_path / "TLBVFI"
        if str(tlbvfi_path) not in sys.path:
            sys.path.insert(0, str(tlbvfi_path))
        
        from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel
        from model.utils import dict2namespace

        global _CURRENT_MODEL, _CURRENT_MODEL_KEY
        cache_key = (model_name, diffusion_steps)
        
        if _CURRENT_MODEL_KEY == cache_key and _CURRENT_MODEL is not None:
            model = _CURRENT_MODEL
        else:
            if _CURRENT_MODEL is not None:
                _CURRENT_MODEL = None
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            model_path = folder_paths.get_full_path("interpolation", model_name)
            if not model_path: raise FileNotFoundError(f"Model file {model_name} not found.")

            config_path = tlbvfi_path / "configs" / "Template-LBBDM-video.yaml"
            with open(config_path, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            nconfig = dict2namespace(config)
            nconfig.model.VQGAN.params.ckpt_path = None
            nconfig.model.BB.params.sample_step = diffusion_steps
            
            model = LatentBrownianBridgeModel(nconfig.model).to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint.get('model', checkpoint))
            model.float().eval()
            _CURRENT_MODEL, _CURRENT_MODEL_KEY = model, cache_key

        image_tensors = images.permute(0, 3, 1, 2).float()
        image_tensors = (image_tensors * 2.0) - 1.0
        
        if len(image_tensors) < 2:
            return io.NodeOutput(images)

        num_pairs = len(image_tensors) - 1
        gui_pbar = ProgressBar(num_pairs)
        output_frames = [image_tensors[0:1]]

        with torch.no_grad():
            for i in tqdm(range(0, num_pairs, batch_size), desc="TLBVFI Interpolating"):
                current_batch_size = min(batch_size, num_pairs - i)
                f1_batch = image_tensors[i : i + current_batch_size].to(device)
                f2_batch = image_tensors[i + 1 : i + 1 + current_batch_size].to(device)
                
                current_frames = [f1_batch, f2_batch]
                for _ in range(times_to_interpolate):
                    temp_frames = [current_frames[0]]
                    for j in range(len(current_frames) - 1):
                        mid_frame = model.sample(current_frames[j], current_frames[j+1], scale=flow_scale, disable_progress=True)
                        mid_frame = torch.nan_to_num(mid_frame, nan=0.0, posinf=1.0, neginf=-1.0).cpu()
                        temp_frames.extend([mid_frame, current_frames[j+1].cpu()])
                    current_frames = temp_frames
                
                for b in range(current_batch_size):
                    for k in range(1, len(current_frames)):
                        output_frames.append(current_frames[k][b:b+1])
                gui_pbar.update(current_batch_size)

        final_tensors = torch.cat(output_frames, dim=0)
        final_tensors = (final_tensors + 1.0) / 2.0
        return io.NodeOutput(final_tensors.clamp(0, 1).permute(0, 2, 3, 1))

class TLBVFIExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TLBVFI_VFI]

async def comfy_entrypoint() -> TLBVFIExtension:
    return TLBVFIExtension()
