# app_cpu.py

import torch
from diffusers import StableDiffusionPipeline
import time  # To measure execution time
import sys
import os

# --- Configuration ---
# Using the standard v1.5 model, suitable for CPU.
# model_id = "runwayml/stable-diffusion-v1-5"
# Or stick with the legacy one if you prefer/have it downloaded
model_id = "sd-legacy/stable-diffusion-v1-5"

output_filename = "img1.png"
prompt ="dr:doom and spiderman fighting in a futuristic city, comic book style, vibrant colors, dynamic action scene, highly detailed, 4k resolution"

# IMPORTANT: Use float32 for CPU compatibility and stability
# float16 is generally optimized for GPUs and may be slower or less stable on CPU
torch_dtype = torch.float32
# --------------------

# 1. Basic PyTorch Check (No CUDA verification needed)
print("="*80)
print("Verifying PyTorch installation...")
try:
    print(f"PyTorch version: {torch.__version__}")
    # Check if this version includes '+cpu'
    if '+cpu' in torch.__version__:
        print("Detected CPU-only PyTorch build.")
    else:
        # Even if a CUDA build is installed, we won't use the GPU
        print("PyTorch build seems to support CUDA, but we will run on CPU only.")
except Exception as e:
    print(f"An error occurred during PyTorch verification: {e}")
    sys.exit(1)

print("PyTorch check complete. Proceeding with CPU execution.")
print("="*80)
print(f"NOTE: Running Stable Diffusion on CPU will be significantly slower than on a GPU.")
print("Please be patient, image generation may take several minutes.")
print("="*80)

# 2. Load the Stable Diffusion Pipeline
print(f"Loading Stable Diffusion pipeline for model: '{model_id}'...")
print(f"Using data type: {torch_dtype} (Recommended for CPU)")

# Record start time for loading
load_start_time = time.time()

try:
    # Load the pipeline specifically requesting float32
    # We do NOT call .to("cuda")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        # If you previously disabled the safety checker for CUDA,
        # you might keep it disabled or re-enable it as needed.
        # safety_checker=None, # Uncomment to disable if desired
    )
    print("Pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    print("This might be due to:")
    print("- Network issues preventing model download.")
    print("- Insufficient RAM or disk space for the model.")
    print("- Incorrect model_id or missing model files.")
    sys.exit(1)

load_end_time = time.time()
print(f"Pipeline loading took: {load_end_time - load_start_time:.2f} seconds.")

# NOTE: No step 3 for moving to GPU

# 4. Generate the Image (on CPU)
print("="*80)
print(f"Generating image for prompt: '{prompt}' (using CPU)...")
print("This is the slow part - please wait patiently!")

# Record start time for generation
gen_start_time = time.time()

try:
    # Run the inference - this happens entirely on the CPU now
    output = pipe(prompt)
    image = output.images[0] # Get the first generated image
    print("Image generation complete.")

    # Optional: Check for NSFW content (if safety checker wasn't disabled)
    if hasattr(pipe, 'safety_checker') and pipe.safety_checker is not None:
        if hasattr(output, 'nsfw_content_detected') and output.nsfw_content_detected:
             if isinstance(output.nsfw_content_detected, list) and any(output.nsfw_content_detected):
                 print("Warning: Potential NSFW content detected in the generated image.")
             elif isinstance(output.nsfw_content_detected, bool) and output.nsfw_content_detected:
                 print("Warning: Potential NSFW content detected in the generated image.")

except Exception as e:
    print(f"Error during image generation: {e}")
    print("This can occur due to:")
    print("- Out of system RAM (CPU inference can use significant RAM).")
    print("- Issues with the model components.")
    print("- Corrupted model download.")
    sys.exit(1)

gen_end_time = time.time()
print(f"Image generation took: {gen_end_time - gen_start_time:.2f} seconds.")

# 5. Save the Image
print(f"Saving generated image to '{output_filename}'...")
try:
    image.save(output_filename)
    print(f"Image successfully saved to: {os.path.abspath(output_filename)}")
except Exception as e:
    print(f"Error saving image: {e}")
    print("Check folder permissions and if the path is valid.")
    sys.exit(1)

print("="*80)
print("Script finished successfully using CPU!")
total_time = time.time() - load_start_time # Calculate total time from loading start
print(f"Total execution time (loading + generation): {total_time:.2f} seconds.")
print("="*80)