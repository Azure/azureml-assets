<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `description.md` is required. -->

MedSAM2 is a breakthrough medical image segmentation foundation model that addresses the critical need for efficient and accurate segmentation in precision medicine. Built upon the Segment Anything Model (SAM) 2.1 architecture, it uniquely bridges the gap between 2D and 3D medical image segmentation, offering a unified solution for both volumetric scans and video data. The model stands out for its ability to reduce manual annotation costs by over 85% while maintaining high accuracy across diverse medical imaging applications.

The model has been extensively trained on over 455,000 3D image-mask pairs and 76,000 annotated video frames, spanning multiple organs, pathologies, and imaging protocols. MedSAM2's versatility has been validated through comprehensive user studies involving 5,000 CT lesions, 3,984 liver MRI lesions, and 251,550 echocardiogram video frames. The model incorporates a novel self-sorting memory bank mechanism for dynamic feature selection, enabling robust performance across different medical imaging contexts. This makes it particularly valuable for clinical workflows in cardiology, oncology, and surgical specialties where precise 3D organ and lesion segmentation is critical but traditionally time-consuming. Integrated with user-friendly interfaces for both local and cloud deployment, MedSAM2 represents a practical solution for supporting efficient, scalable, and high-quality segmentation in both research and healthcare environments.

### Model Architecture
MedSAM2 adopts the SAM2 network architecture with four main components:
- Image encoder (Hierarchical Vision Transformer)
- Memory attention module
- Prompt encoder
- Mask decoder

The model uses SAM2.1-Tiny variant with full-model fine-tuning approach, optimized for performance with fewer parameters.

### Sample inputs and outputs (for real time inference)
Input:
```bash
data = {
  "input_data": {
    "nii_image": ["base64_encoded_nifti_file"],
    "bbox": [[50.0, 50.0, 150.0, 150.0]],
    "key_slice_idx": [5],
    "dicom_window": [[0.0, 255.0]],
    "slice_offset": [0]
  }
}
```

**Output Sample**
```json
{
  "mask": [[[0, 0, 1, 1, 0], [0, 1, 1, 1, 0]], ...],
  "metadata": {
    "shape": [512, 512, 100],
    "key_slice_index": 5,
    "adjusted_key_slice_index": 5,
    "bbox": [50.0, 50.0, 150.0, 150.0],
    "dicom_window": [0.0, 255.0],
    "spacing": [1.0, 1.0, 1.0],
    "origin": [0.0, 0.0, 0.0]
  }
}
```

**Output Processing Example:**
```python
def process_medsam2_output(result, file_path):
    """Process MedSAM2 segmentation output and extract key metrics."""
    if 'error' in result:
        return {"error": result['error'], "status": result.get('status')}
    
    # Handle different response formats
    if isinstance(result, list) and len(result) > 0:
        result = result[0]
    
    processed_result = {"file_path": file_path}
    
    if 'mask' in result:
        mask = np.array(result['mask'])
        
        # Segmentation statistics
        total_voxels = mask.size
        segmented_voxels = np.sum(mask > 0)
        segmentation_ratio = segmented_voxels / total_voxels * 100
        
        processed_result.update({
            "mask_shape": mask.shape,
            "mask_dtype": str(mask.dtype),
            "unique_values": np.unique(mask).tolist(),
            "total_voxels": int(total_voxels),
            "segmented_voxels": int(segmented_voxels),
            "segmentation_ratio": round(segmentation_ratio, 2),
            "segmentation_successful": True
        })
    
    if 'metadata' in result:
        metadata = result['metadata']
        processed_result.update({
            "image_shape": metadata.get('shape'),
            "key_slice_index": metadata.get('key_slice_index'),
            "adjusted_key_slice_index": metadata.get('adjusted_key_slice_index'),
            "bbox": metadata.get('bbox'),
            "dicom_window": metadata.get('dicom_window'),
            "spacing": metadata.get('spacing'),
            "origin": metadata.get('origin')
        })
    
    return processed_result
```