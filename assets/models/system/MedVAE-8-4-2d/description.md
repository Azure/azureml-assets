MedVAE is a family of large-scale, generalizable 2D and 3D variational autoencoders (VAEs) designed to address critical efficiency and storage challenges in medical imaging. Trained on over one million images across multiple modalities and anatomical regions, MedVAE excels at encoding high-resolution medical images into compact, feature-rich latent representations. This process allows downstream computer-aided diagnosis (CAD) models to operate on these small vectors instead of large image files, leading to significant performance benefits—up to a 70x improvement in model throughput and a 512x reduction in data storage requirements.

The core value of MedVAE lies in its ability to preserve clinically-relevant features within these downsized latent representations. This ensures that the efficiency gains do not come at the cost of diagnostic accuracy. By providing a powerful feature extraction mechanism, MedVAE enables the development of faster, more scalable, and cost-effective medical AI solutions, from rapid image retrieval systems to complex diagnostic classifiers, making it an essential tool for both clinical research and healthcare environments.

### Model Architecture
MedVAE is built on a Variational Autoencoder (VAE) architecture, consisting of an encoder that compresses images into a latent space and a decoder that reconstructs images from that space. The model undergoes a novel two-stage training process:
- **Stage 1**: The autoencoder is trained on a large-scale dataset of over one million medical images to learn a robust general representation.
- **Stage 2**: The model is fine-tuned using a consistency loss guided by BiomedCLIP embeddings. This stage refines the latent space to ensure that critical, clinically-relevant features are accurately captured and preserved.

This model is the **2D MedVAE (f=64, C=4) variant** variant, which downsamples the input image by a factor of 8 in each spatial dimension and produces a four-channel latent representation.

### Sample inputs and outputs (for real time inference)
Input:
The model endpoint expects a JSON payload containing the image path (as a URL/local path or base64-encoded string) and an optional flag to request the decoded image.

```bash
# Example with curl, sending an image URL
API_KEY="YOUR_API_KEY"
ENDPOINT_URL="YOUR_ENDPOINT_URL"
DEPLOYMENT_NAME="DEPLOYMENT_NAME"
IMAGE_URL="sample_chest_xray.png"

curl -X POST "$ENDPOINT_URL" \
-H "Authorization: Bearer $API_KEY" \
-H "Content-Type: application/json" \
-H "azureml-model-deployment: $DEPLOYMENT_NAME" \
-d '{
  "input_data": {
    "columns": ["image_path", "decode"],
    "data": [
      [
        "'"$IMAGE_URL"'",
        true
      ]
    ]
  }
}'
```

**Output Sample**
The model returns a JSON object containing the latent vector and, if requested, the decoded image reconstruction.

```json
{
  "predictions": [
    {
      "latent": [[-0.52, 0.88, ...], [-0.43, 0.91, ...], ...],
      "decoded": [[15, 16, 15, ...], [16, 17, 16, ...], ...]
    }
  ]
}
```

**Output Processing Example:**
This Python function shows how to process the JSON response to save the latent vector as a `.npy` file.

```python
import numpy as np
from PIL import Image
import os

def process_medvae_output(result, output_path):
    """Process MedVAE output and save the latent vector and decoded image."""
    if not output_path:
        return

    # Handle nested response format
    if 'predictions' in result:
        result = result['predictions']
    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    base_output_path = os.path.splitext(output_path)[0]

    if "latent" in result:
        latent_path = f"{base_output_path}_latent.npy"
        np.save(latent_path, np.array(result['latent']))
        print(f"Latent vector saved to: {latent_path}")

    if "decoded" in result:
        decoded_image_path = f"{base_output_path}_decoded.png"
        decoded_array = np.array(result['decoded'])
        
        # Transpose if channel-first (e.g., 1xHxW -> HxWx1)
        if decoded_array.ndim == 3 and decoded_array.shape[0] in [1, 3]:
             decoded_array = np.transpose(decoded_array, (1, 2, 0))
        
        # Normalize to 0-255 for image saving
        decoded_array = ((decoded_array - decoded_array.min()) / 
                         (decoded_array.max() - decoded_array.min()) * 255)
        
        img = Image.fromarray(decoded_array.astype(np.uint8).squeeze())
        img.save(decoded_image_path)
        print(f"Decoded image saved to: {decoded_image_path}")

```

## Data and Resource Specification for Deployment
* **Supported Data Input Format** 

1. **Input Format**: The model accepts 2D medical images in standard formats like PNG, JPEG, etc.

2. **Input Methods**: The model supports both:
   - **Base64-encoded images**: Local image files encoded as base64 strings.
   - **Direct URLs**: Publicly accessible URLs pointing to remote image files.

3. **Input Schema Requirements**:
   - `image_path`: (string) A URL to an image or a base64-encoded image string.
   - `decode`: (boolean, optional) If `true`, the model returns both the latent vector and the reconstructed image. If `false` or omitted, it returns only the latent vector.

4. **Supported Medical Image Types and Processing Specifications**:
   - **Modalities**: Optimized for 2D images such as X-ray and full-field digital mammograms.
   - **Processing**: Input images are automatically transformed and resized by the model for inference.
   - **Output**: The primary output is a 2D latent vector.