# ComfyUI-TLBVFI

A LLM coded node pack for ComfyUI that provides video frame interpolation using the **TLB-VFI** model.

This is a wrapper for the [TLB-VFI: Temporal-Aware Latent Brownian Bridge Diffusion for Video Frame Interpolation](https://github.com/ZonglinL/TLBVFI) project, allowing for integration into ComfyUI.

## Features
-   **High-Quality Interpolation**: Leverages a powerful latent diffusion model to generate smooth and detailed in-between frames.
-   **Configurable Interpolation Steps**: Easily double, quadruple, or octuple your frame rate by adjusting the `times_to_interpolate` setting.

---

## ⚙️ Installation

Please follow these steps carefully to ensure the node is set up correctly.

### Step 1: Install the Custom Node
If you are using the [ComfyUI-Manager](https://github.com/Comfy-Org/ComfyUI-Manager), you can install this node from there.

Alternatively, you can install it manually by cloning this repository into your `ComfyUI/custom_nodes/` directory...

# Navigate to your ComfyUI custom_nodes directory
```bash
cd ComfyUI/custom_nodes/

# Clone this repository
git clone https://github.com/BobRandomNumber/ComfyUI-TLBVFI.git
```

### Step 2: Install Dependencies

```bash
# Navigate into the newly created custom node directory
cd ComfyUI/custom_nodes/ComfyUI-TLBVFI/

pip install -r requirements.txt
```

### Step 3: Download the Pre-trained Model
Only one model file is required to run the interpolation.
- **Full Model:** `vimeo_unet.pth`

Download the file from the official Hugging Face repository:
-   **[ucfzl/TLBVFI on Hugging Face](https://huggingface.co/ucfzl/TLBVFI/tree/main)**

### Step 4: Place Model in the `interpolation` Folder
This node looks for models in the `ComfyUI/models/interpolation/` directory.

1.  Place the downloaded `vimeo_unet.pth` file into this folder.

For better organization, you are welcome to create a subdirectory. The node will find the model automatically.

**Example Folder Structure:**
```
ComfyUI/
└── models/
    └── interpolation/
        └── tlbvfi_models/
            └── vimeo_unet.pth
```

> **For advanced users:** If you prefer to store models elsewhere, you can add a path to your `extra_model_paths.yaml` file and assign it the type `interpolation`.

### Step 5: Restart ComfyUI
After completing all the steps, **restart ComfyUI**.

---

## 🚀 Usage

1.  In ComfyUI, add the **TLBVFI Frame Interpolation** node. You can find it by right-clicking and searching, or under the `frame_interpolation/TLBVFI` category.
2.  Connect a batch of loaded images (e.g., from a `Load Video` or `Load Image Batch` node) to the `images` input.
3.  Select the correct model from the dropdown menu:
    *   **`model_name`**: Choose `vimeo_unet.pth` (or `tlbvfi_models/vimeo_unet.pth` if you used a subfolder).
4.  Adjust **`times_to_interpolate`** to control how many new frames are generated between each pair of original frames:
    *   `1`: **Doubles** the frame count (1 new frame).
    *   `2`: **Quadruples** the frame count (3 new frames).
    *   `3`: **8x** the frame count (7 new frames).
5.  Connect the output `IMAGE` to a `Save Image` or `Preview Image` node to see your interpolated sequence.

---

## 🧠 How It Works

This node uses a two-stage **latent diffusion** process:

1.  **VQGAN (Autoencoder)**: First, the VQGAN model takes your full-resolution input frames and compresses them into a small, efficient "latent space."
2.  **UNet (Diffusion Model)**: The core interpolation logic happens in this latent space. The UNet takes the compressed representations of the start and end frames and generates the latent representation for the frame in between.
3.  **VQGAN (Decoder)**: Finally, the VQGAN's decoder takes the newly generated latent and reconstructs it back into a full-resolution, detailed image.

This approach is highly efficient and allows for the generation of high-quality, temporally consistent frames.

---

## 🙏 Acknowledgements and Citation

This node is a wrapper implementation for ComfyUI. All credit for the model architecture, training, and research goes to the original authors of TLB-VFI. If you use this model in your research, please cite their work.

-   **Original GitHub Repository**: [https://github.com/ZonglinL/TLBVFI](https://github.com/ZonglinL/TLBVFI)
-   **Project Page**: [https://zonglinl.github.io/tlbvfi_page/](https://zonglinl.github.io/tlbvfi_page/)

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
