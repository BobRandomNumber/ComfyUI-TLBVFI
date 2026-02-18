# ComfyUI-TLBVFI

A node pack for ComfyUI that provides video frame interpolation using the **TLB-VFI** model.

This is a wrapper for the [TLB-VFI: Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation](https://github.com/ZonglinL/TLBVFI) project.

## Features
- **Zero-Dependency**: All non-standard requirements (CuPy, PyTorch-Lightning, etc.) have been removed or replaced with native implementations.
- **Efficient Batching**: Supports processing multiple frame pairs simultaneously.

## ⚙️ Installation

### Step 1: Install the Custom Node
Clone this repository into your `ComfyUI/custom_nodes/` directory:

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/BobRandomNumber/ComfyUI-TLBVFI.git
```

### Step 2: Download the Pre-trained Model
Download `vimeo_unet.pth` from the official repository:
- **[ucfzl/TLBVFI on Hugging Face](https://huggingface.co/ucfzl/TLBVFI/tree/main)**

### Step 3: Place Model in the `interpolation` Folder
Place the downloaded `.pth` file into `ComfyUI/models/interpolation/`.

```
ComfyUI/
└── models/
    └── interpolation/
        └── vimeo_unet.pth
```

## 🚀 Usage

1. Add the **TLBVFI Frame Interpolation** node from the `frame_interpolation/TLBVFI` category.
2. Select the correct model from the **`model_name`** dropdown.
3. **`times_to_interpolate`**: Sets how many new frames are generated between pairs (1 = double FPS, 2 = quadruple, etc.).
4. **`diffusion_steps`**: Controls the refinement quality. Higher values (e.g., 20-50) improve quality at the cost of speed.
5. **`batch_size`**: Number of pairs to process at once. Increase if you have sufficient VRAM for a speed boost.
6. **`flow_scale`**: Resolution for motion analysis. Use `0.5` for most videos; lower values handle fast motion better.

## 🧠 How It Works

1. **VQGAN**: Compresses input frames into a latent space.
2. **UNet (Brownian Bridge)**: Operates in latent space to diffuse and generate the in-between representation using a reverse diffusion process.
3. **VQGAN Decoder**: Reconstructs the generated latent back into a full-resolution image.

## 🙏 Acknowledgements

All credit for the architecture and research goes to the original authors of TLB-VFI.

- **Original GitHub**: [ZonglinL/TLBVFI](https://github.com/ZonglinL/TLBVFI)
- **Project Page**: [https://zonglinl.github.io/tlbvfi_page/](https://zonglinl.github.io/tlbvfi_page/)

```bibtex
@article{lyu2025tlbvfitemporalawarelatentbrownian,
      title={TLB-VFI: Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation}, 
      author={Zonglin Lyu and Chen Chen},
      year={2025},
      eprint={2507.04984},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```
