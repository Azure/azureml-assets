<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `description.md` is required. -->

Biomedical image analysis is fundamental for biomedical discovery in cell biology, pathology, radiology, and many other biomedical domains. 3D medical images such as CT and MRI play unique roles in clinical practices. MedImageParse 3D is a foundation model for imaging parsing that can jointly conduct segmentation, detection, and recognition for 3D medical images including CT and MRI. Through joint learning, we can improve accuracy for individual tasks and enable novel applications such as segmenting relevant objects in an image through a text prompt, rather than requiring users to laboriously specify the bounding box for each object.

MedImageParse 3D was trained on a large dataset comprising triples of image, segmentation mask, and textual description. It takes in 3D medical image volume with a text prompt about the target object type (e.g. pancreas in CT), and outputs the corresponding segmentation mask in 3D volume the same shape as the input image. MedImageParse 3D is also able to identify invalid user inputs describing objects that do not exist in the image. MedImageParse 3D can perform object detection, which aims to locate a specific object of interest, including objects with irregular shapes or of small size.

Traditional segmentation models do segmentation alone, requiring a fully supervised mask during training and typically need either manual bounding boxes or automatic proposals at inference if multiple objects are present. Such model doesn’t inherently know which object to segment unless trained specifically for that class, and it can’t take a text query to switch targets. MedImageParse 3D can segment via text prompts describing the object without needing a user-drawn bounding box. This semantic prompt-based approach lets it parse the image and find relevant objects anywhere in the image.

In summary, MedImageParse 3D shows potential to be a building block for an all-in-one tool for biomedical image analysis by jointly solving segmentation, detection, and recognition. It is broadly applicable to different 3D image modalities through text prompting, which may pave a future path for efficient and accurate image-based biomedical discovery when built upon and integrated into an application.

### Model Architecture

MedImageParse 3D is built upon BiomedParse with the BoltzFormer architecture, optimized for locating small objects in 3D images. Leveraging Boltzmann attention sampling mechanisms, it excels at identifying subtle patterns corresponding to biomedical terminologies, as well as extracting contextually relevant information from dense scientific texts. The model is pre-trained on vast 3D medical image datasets, allowing it to generalize across various biomedical domains with high accuracy.

### Sample inputs and outputs (for real time inference)

**Input**

```python
import base64
data = {
    "input_data": {
        "columns": ["image", "text"],
        "index": [0],
        "data": [
            [
                # Base64-encoded .nii.gz data:
                base64.b64encode(open("./examples/example.nii.gz", "rb").read()).decode("utf-8"),
                # Example text/string input:
                "pancreas"
            ]
        ]
    }
}
```

- `"columns"` describes the types of inputs your model expects (in this case, `"image"` and `"text"`).
- `"data"` is where you actually provide the values: the first element is the Base64-encoded NIfTI, and the second is a text parameter (e.g., `"pancreas"`).

**Output**

```json
[
  {
    "nifti_file": "{\"data\":\"<BASE64_ENCODED_BYTES>\"}"
  }
]
```

The field `"<Base64EncodedNifti>"` contains raw binary NIfTI data, encoded in Base64.

The provided function `decode_base64_to_nifti` handles the decoding logic:

```python
import json
import base64
import tempfile
import nibabel as nib

def decode_base64_to_nifti(base64_string: str) -> nib.Nifti1Image:
    """
    Decode a Base64 string back to a NIfTI image.
  
    The function expects `base64_string` to be a JSON string
    of the form: '{"data": "<Base64EncodedBytes>"}'.
    """
    # Convert the 'nifti_file' string to a Python dict, then extract the 'data' field
    base64_string = json.loads(base64_string)["data"]
  
    # Decode Base64 string to raw bytes
    byte_data = base64.b64decode(base64_string)
  
    # Write these bytes to a temporary file and load as a NIfTI image
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
        temp_file.write(byte_data)
        nifti_image = nib.load(temp_file.name)
  
    # Return the voxel data as a Numpy array
    return nifti_image.get_fdata()
```

The output can be parsed using:

```python
import json

# Suppose `response` is the raw byte response from urllib
response_data = json.loads(response)

# Extract the JSON-stringified NIfTI
nifti_file_str = response_data[0]["nifti_file"]

# Decode to get the NIfTI volume as a Numpy array
segmentation_array = decode_base64_to_nifti(nifti_file_str)
print(segmentation_array.shape)  # e.g., (512, 512, 128)
```

Optionally, the `plot_segmentation_masks` helper function shows slices of the 3D array if they contain non-zero content:

```python
import matplotlib.pyplot as plt 

def plot_segmentation_masks(segmentation_masks):
    """
    Plot each axial slice (z-slice) of the segmentation if it contains a non-zero mask.
    """
    index = 1
    plt.figure(figsize=(15, 15))
    for i in range(segmentation_masks.shape[2]):
        if segmentation_masks[:, :, i].sum() > 0:
            plt.subplot(4, 4, index)
            plt.imshow(segmentation_masks[:, :, i], cmap='gray')
            plt.axis('off')
            index += 1
    plt.show()
```
