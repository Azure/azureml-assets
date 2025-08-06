Hibou-L is a foundational vision transformer developed for digital pathology, designed to generate high-quality feature representations from histology image patches. These representations can be leveraged for a range of downstream tasks, including classification, segmentation, and detection.

Built on the ViT-B/14 architecture, Hibou-L is a digital pathology foundation model pretrained on a 1.2B image private dataset using the DINOv2 framework. The model processes 224 × 224 input patches and generates high-quality feature representations for downstream histopathological tasks.

### Evaluation Results
To understand the capabilities of Hibou-L, we evaluated it across a range of public digital pathology benchmarks. The model was tested at both patch-level and slide-level granularity to assess its generalization and diagnostic utility across different cancer types and tissue modalities.

| Category | Benchmark | Hibou-L |
|--|--|--|
| Patch-level | CRC-100K | 96.6 |
|  | PCAM | 95.3 |
|  | MHIST | 85.8 |
|  | MSI-CRC | 79.3 |
|  | MSI-STAD | 82.9 |
|  | TIL-DET | 94.2 |
| Slide-level | BRCA | 94.6 |
|  | NSCLC | 96.9 |
|  | RCC | 99.6 |

### Sample inputs and outputs (for real time inference)

**Single PDB Input:**
```bash
data = {
  "input_data": {
    "columns": ["image"],
    "data": [
      ["base64_encoded_image_string"]
    ]
  }
}
```

**Multiple PDB Input:**
```bash
data = {
  "input_data": {
    "columns": ["image"], 
    "data": [
      ["base64_encoded_image_string_1"],
      ["base64_encoded_image_string_2"]
    ]
  }
}
```

**Output Sample:**
```json
[
  {
    "image_features": [2.6749861240386963, -0.7507642507553101, 0.2108164280653, ...]
  }
]
```

**Output Processing Example:**
```python
def process_hibou_predictions(result):
    """Process Hibou-L embedding predictions."""
    if not result:
        print("No predictions found")
        return

    # Handle the response format: [{'image_features': [embedding_list]}]
    if isinstance(result, list) and len(result) > 0:
        first_result = result[0]
        if isinstance(first_result, dict) and 'image_features' in first_result:
            embeddings = first_result['image_features']
            embedding_dim = len(embeddings)
            print(f"Received embeddings with dimension: {embedding_dim}")
            
            # Calculate statistics
            embedding_array = np.array(embeddings)
            print(f"Embedding statistics:")
            print(f"  - Mean: {np.mean(embedding_array):.4f}")
            print(f"  - Std: {np.std(embedding_array):.4f}")
            print(f"  - Min: {np.min(embedding_array):.4f}")
            print(f"  - Max: {np.max(embedding_array):.4f}")
            
            return embeddings
        else:
            print(f"Unexpected result format - missing 'image_features' key")
            print(f"Available keys: {list(first_result.keys()) if isinstance(first_result, dict) else 'Not a dict'}")
    else:
        print(f"Unexpected result format: {type(result)}")
    
    return None

def visualize_embeddings_results(original_image, embeddings, save_path=None):
    """Visualize the embedding results with distribution plot."""
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(original_image)
    embedding_dim = len(embeddings) if embeddings else 0
    plt.title(f'Processed - Embedding dim: {embedding_dim}')
    plt.axis('off')
    
    # Show embedding distribution
    plt.subplot(1, 3, 3)
    if embeddings:
        embedding_array = np.array(embeddings)
        plt.hist(embedding_array, bins=50, alpha=0.7, color='blue')
        plt.title(f'Embedding Distribution\nMean: {np.mean(embedding_array):.3f}\nStd: {np.std(embedding_array):.3f}')
        plt.xlabel('Embedding Value')
        plt.ylabel('Frequency')
    else:
        plt.text(0.5, 0.5, 'No embeddings', ha='center', va='center')
        plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
```

## Data and Resource Specification for Deployment
* **Supported Data Input Format** 
1. **Input Format**: The model accepts histopathology images in png. Images can be provided as base64-encoded image strings.

2. **Output Format**: The model generates dense embedding vectors representing the visual features of histopathology image in 768-dimensional feature vectors.

3. **Data Sources and Technical Details**: For comprehensive information about training datasets, model architecture, and validation results, refer to the [official hibou repository](https://github.com/HistAI/hibou/tree/main)