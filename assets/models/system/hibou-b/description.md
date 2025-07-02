<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `description.md` is required. -->

ATOMICA is a hierarchical geometric deep learning model trained on over 2.1 million molecular interaction interfaces. It represents interaction complexes using an all-atom graph structure, where nodes correspond to atoms or grouped chemical blocks, and edges reflect both intra- and intermolecular spatial relationships. The model uses SE(3)-equivariant message passing to ensure that learned embeddings are invariant to rotations and translations of molecular structures. The architecture produces embeddings at multiple scales—atom, block, and graph—that capture fine-grained structural detail and broader functional motifs. [Project Website](https://zitniklab.hms.harvard.edu/projects/ATOMICA/)

The model has been pretrained on 2,105,703 molecular interaction interfaces from the Protein Data Bank and Cambridge Structural Database, spanning multiple interaction types including protein-small molecule, protein-ion, small molecule-small molecule, protein-protein, protein-peptide, protein-RNA, protein-DNA, and nucleic acid-small molecule complexes. The pretraining strategy involves denoising transformations (rotation, translation, torsion) and masked block-type prediction, enabling the model to learn chemically grounded, transferable features. ATOMICA supports downstream tasks via plug-and-play adaptation with task-specific heads, including binding site prediction and protein interface fingerprinting. [Project Website](https://zitniklab.hms.harvard.edu/projects/ATOMICA/) | [GitHub](https://github.com/mims-harvard/ATOMICA/tree/main)



### Model Architecture
ATOMICA employs a hierarchical geometric deep learning architecture with the following key components:
- An all-atom graph representation of molecular complexes where nodes correspond to atoms or chemical blocks
- SE(3)-equivariant message passing to ensure rotational and translational invariance
- Multi-scale embeddings at atom, block, and graph levels
- Pretraining via denoising transformations and masked block-type prediction

### Sample inputs and outputs (for real time inference)

**Single PDB Input:**
```bash
data = {
  "input_data": {
    "columns": ["pdb_data"],
    "index": [0],
    "data": [
      ["https://path/to/local/protein.jsonl.gz"]
    ]
  }
}
```

**Multiple PDB Input:**
```bash
data = {
  "input_data": {
    "columns": ["pdb_data"],
    "index": [0, 1],
    "data": [
      ["https://path/to/local/protein1.jsonl.gz"],
      ["https://path/to/local/protein2.jsonl.gz"]
    ]
  }
}
```

**Output Sample:**
```json
{
  "predictions": [
    {
      "graph_embedding": [0.123456, -0.234567, ...],
      "block_embedding": [[0.345678, -0.456789, ...], ...],
      "atom_embedding": [[0.567890, -0.678901, ...], ...]
    }
  ]
}
```

**Output Processing Example:**
```python
def process_predictions(predictions, is_batch=False):
    """Process predictions in a consistent way."""
    if not predictions:
        print("No predictions found")
        return

    # Extract predictions from response
    if isinstance(predictions, dict) and 'predictions' in predictions:
        predictions = predictions['predictions']

    # Handle both single and batch predictions
    if is_batch:
        if isinstance(predictions, list):
            # Multiple predictions
            for i, pred in enumerate(predictions, 1):
                print(f"\nPDB {i}:")
                _display_embeddings(pred)
        elif isinstance(predictions, dict):
            # Single prediction in batch format
            _display_embeddings(predictions)
        else:
            print(f"Error: Unexpected predictions type: {type(predictions)}")
    else:
        # Single prediction format
        if isinstance(predictions, list) and len(predictions) > 0:
            # Single prediction in list
            _display_embeddings(predictions[0])
        elif isinstance(predictions, dict):
            # Direct dictionary format
            _display_embeddings(predictions)
        else:
            print(f"Error: Invalid prediction format: {type(predictions)}")

def _display_embeddings(pred):
    """Display embeddings in a concise format."""
    if not isinstance(pred, dict):
        print(f"Error: Invalid prediction format: {type(pred)}")
        return
        
    for embed_type in ["graph_embedding", "block_embedding", "atom_embedding"]:
        if embed_type in pred:
            embedding = pred[embed_type]
            if isinstance(embedding, list):
                arr = np.array(embedding)
                print(f"\n{embed_type}:")
                print(f"Shape: {arr.shape}")
                print(f"Mean: {arr.mean():.6f}, Std: {arr.std():.6f}")
                print(f"Range: [{arr.min():.6f}, {arr.max():.6f}]")
        else:
            print(f"\nWarning: {embed_type} not found in predictions")
            print("Available keys:", list(pred.keys()) if isinstance(pred, dict) else "Not a dictionary")
```

## Data and Resource Specification for Deployment
* **Supported Data Input Format** 
Text to text