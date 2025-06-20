# Text-to-Image Diffusion Model Training

##  Technical Overview

This implementation provides a complete training pipeline for text-conditional diffusion models using the **Denoising Diffusion Probabilistic Models (DDPM)** framework. The architecture follows the **Stable Diffusion** paradigm and includes three key components:

- A Variational Autoencoder (VAE) for encoding/decoding images
- A CLIP text encoder for text conditioning
- A UNet denoising network with cross-attention

---

##  Mathematical Framework

### Forward Process

The forward noising process gradually corrupts the image:

$$
q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \cdot x_{t-1}, \beta_t \cdot \mathbf{I})
$$

### Reverse Process

The model learns the reverse denoising distribution:

$$
p_\theta(x_{t-1} \mid x_t, c) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, c), \Sigma_\theta(x_t, t, c))
$$

where:
- \( c \): text conditioning (prompt embedding)
- \( \theta \): learnable parameters

---

##  Architecture Components

1. **VAE Encoder/Decoder**  
   Compresses images:  
   ℝ⁶⁴×⁶⁴×³ → ℝ³²×³²×⁴

2. **CLIP Text Encoder**  
   Converts text prompts into embeddings:  
   ℝ⁷⁷×⁵¹²

3. **UNet Denoising Network**  
   Predicts noise: εθ(zₜ, t, c) with cross-attention conditioning



---

##  Training Objective

The simplified DDPM loss is:

$$
\mathcal{L} = \mathbb{E}_{z_0, \epsilon \sim \mathcal{N}(0, \mathbf{I}), t} \left[ \left\| \epsilon - \epsilon_\theta(z_t, t, c) \right\|^2 \right]
$$

with:

$$
z_t = \sqrt{\bar{\alpha}_t} \cdot z_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon
$$

---

## Dataset Processing

- **Dataset**: CelebA (200k+ celebrity face images)
- **Text Labels**: Generated from facial attributes (e.g., “a person with black hair and sunglasses”)
- **Preprocessing**: Resize to 64×64, normalize to range \([-1, 1]\)

---

## Network Configuration

```python
UNet2DConditionModel:
    sample_size: 32 × 32 (latent space)
    in_channels: 4
    out_channels: 4
    block_channels: [64, 128, 256]
    cross_attention_dim: 512
    timesteps: 1000


Text → CLIP → Text Embeddings (512d)
                     ↓
Image → VAE → Latents → Add Noise → UNet → Predicted Noise
         ↑                           ↑
    Decode ← Denoised Latents ← DDPM Sampling
```


##  Training Status

> ⚠ **Note:** Due to hardware limitations (AMD RX570 with limited VRAM), full model training could not be completed on the current setup.  
> However, the codebase is **fully functional** and has been tested for compatibility with PyTorch and Hugging Face Diffusers.

###  Future Training Plan:

-  Reduce batch size to **1–2 samples**
-  Limit training steps to **50–100 per epoch**
-  Train for **2–3 initial epochs**
-  Utilize **gradient checkpointing** for memory efficiency
-  Optionally reduce model/latent resoluti


