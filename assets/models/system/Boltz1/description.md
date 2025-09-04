Boltz-1 is a breakthrough biomolecular structure prediction foundation model that represents the first fully open-source model to approach AlphaFold3 accuracy. Built to democratize access to state-of-the-art protein structure prediction, Boltz-1 addresses the critical need for accurate, accessible, and efficient biomolecular interaction modeling in drug discovery, structural biology, and biotechnology research.

The model excels at predicting complex biomolecular structures including protein-protein interactions, protein-ligand complexes, and multi-chain assemblies. Boltz-1 incorporates advanced diffusion-based sampling techniques with configurable recycling steps and sophisticated MSA (Multiple Sequence Alignment) processing, enabling robust performance across diverse biomolecular prediction tasks. The model's open-source nature under MIT license makes it freely available for both academic and commercial applications, significantly lowering barriers to entry for computational structural biology.

Developed by the team behind the Boltz foundation models, Boltz-1 has been designed with practical deployment in mind, supporting both local inference and cloud-based scaling. The model represents a significant advancement in making high-quality biomolecular structure prediction accessible to the broader scientific community, from individual researchers to pharmaceutical companies developing novel therapeutics.

### Model Architecture
Boltz-1 adopts a sophisticated diffusion-based architecture with four main components:
- **Pairformer Encoder**: Advanced attention-based sequence and structure representation learning
- **MSA Module**: Multiple sequence alignment processing with configurable subsampling strategies
- **Diffusion Process**: Iterative structure refinement using learned diffusion dynamics
- **Confidence Prediction**: Integrated confidence scoring for predicted structures

The model uses optimized sampling parameters with configurable diffusion steps, recycling iterations, and step scaling for balancing prediction quality and computational efficiency.

### Sample inputs and outputs (for real time inference)

**Input (FASTA Format)**:
```json
{
  "input_data": {
    "input_type": "fasta",
    "content": ">protein_A\nMKVLWAAPSKGVTVADAAGAEAKKLVLGDSLSSGKKLEKADPAKPLKPAAR",
    "model_version": "boltz1",
    "output_format": "mmcif",
    "sampling_steps": 200,
    "diffusion_samples": 1,
    "recycling_steps": 3,
    "step_scale": 1.638
  }
}
```

**Input (Protein-Ligand Complex)**:
```json
{
  "input_data": {
    "input_type": "yaml",
    "content": "version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: \"MKVLWAAPSKGVTVADAAGAEAKKLVLGDSLSSGKKLEKADPAKPLKPAAR\"\n  - ligand:\n      id: B\n      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'\nproperties:\n  - affinity:\n      binder: B",
    "model_version": "boltz1",
    "diffusion_samples": 5,
    "affinity_mw_correction": false
  }
}
```

**Output Sample**:
```json
{
  "predicted_structures": [
    {
      "filename": "structure_1_boltz1.mmcif",
      "content": "# mmCIF structure data with atomic coordinates and metadata"
    }
  ],
  "confidence_scores": [
    {
      "filename": "confidence_scores.json",
      "data": {
        "overall_confidence": 0.85,
        "per_residue_confidence": [0.8, 0.85, 0.9, 0.88],
        "parameters_used": {
          "sampling_steps": 200,
          "diffusion_samples": 1,
          "recycling_steps": 3
        }
      }
    }
  ],
  "affinity_scores": {
    "binding_affinity": -8.5,
    "confidence": 0.75,
    "mw_correction_applied": false
  },
  "metadata": {
    "model_version": "boltz1",
    "status": "prediction_completed",
    "prediction_timestamp": 1693747200,
    "computation_parameters": {
      "sampling_steps": 200,
      "diffusion_samples": 1,
      "recycling_steps": 3,
      "output_format": "mmcif"
    }
  }
}
```

**Sample Inference**

```python
import requests
import json

# Endpoint configuration
ENDPOINT_URL = "https://your-endpoint.westus.inference.ml.azure.com/score"
API_KEY = "your_api_key_here"
DEPLOYMENT_NAME = "boltz1-deployment"

# Prepare payload
payload = {
    "input_data": {
        "columns": ["payload"],
        "index": [0],
        "data": [json.dumps({
            "input_type": "fasta",
            "content": ">test_protein\nMKVLWAAPSKGVTVADAAGAEAKKLVLGDSLSSGKKLEKADPAKPLKPAAR",
            "model_version": "boltz1",
            "output_format": "mmcif",
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "recycling_steps": 3
        })]
    }
}

# Send request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
    "azureml-model-deployment": DEPLOYMENT_NAME
}

response = requests.post(ENDPOINT_URL, headers=headers, json=payload, timeout=600)
result = response.json()
```

**cURL Request:**
```bash
curl -X POST "https://your-endpoint.westus.inference.ml.azure.com/score" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -H "azureml-model-deployment: boltz1-deployment" \
  -d '{
    "input_data": {
      "columns": ["payload"],
      "index": [0],
      "data": ["{\"input_type\": \"fasta\", \"content\": \">test_protein\\nMKVLWAAPSKGVTVADAAGAEAKKLVLGDSLSSGKKLEKADPAKPLKPAAR\", \"model_version\": \"boltz1\", \"output_format\": \"mmcif\", \"sampling_steps\": 200, \"diffusion_samples\": 1, \"recycling_steps\": 3}"]
    }
  }'
```

#### 2. Protein-Ligand Complex with Affinity Prediction

**Python Request:**
```python
yaml_content = """version: 1
sequences:
  - protein:
      id: A
      sequence: "MKVLWAAPSKGVTVADAAGAEAKKLVLGDSLSSGKKLEKADPAKPLKPAAR"
  - ligand:
      id: B
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
constraints:
  - pocket:
      binder: B
      contacts: [[A, 10], [A, 15], [A, 20]]
      max_distance: 6.0
properties:
  - affinity:
      binder: B"""

payload = {
    "input_data": {
        "columns": ["payload"],
        "index": [0],
        "data": [json.dumps({
            "input_type": "yaml",
            "content": yaml_content,
            "model_version": "boltz1",
            "output_format": "mmcif",
            "sampling_steps": 200,
            "diffusion_samples": 5,
            "affinity_mw_correction": False,
            "sampling_steps_affinity": 200,
            "diffusion_samples_affinity": 5
        })]
    }
}
```

**cURL Request (save YAML to file first):**
```bash
# Save YAML content to file
cat > protein_ligand.yaml << 'EOF'
version: 1
sequences:
  - protein:
      id: A
      sequence: "MKVLWAAPSKGVTVADAAGAEAKKLVLGDSLSSGKKLEKADPAKPLKPAAR"
  - ligand:
      id: B
      smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'
constraints:
  - pocket:
      binder: B
      contacts: [[A, 10], [A, 15], [A, 20]]
      max_distance: 6.0
properties:
  - affinity:
      binder: B
EOF

# Create JSON payload
YAML_CONTENT=$(cat protein_ligand.yaml | sed 's/"/\\"/g' | tr '\n' '\\n')

curl -X POST "https://your-endpoint.westus.inference.ml.azure.com/score" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key_here" \
  -H "azureml-model-deployment: boltz1-deployment" \
  -d "{
    \"input_data\": {
      \"columns\": [\"payload\"],
      \"index\": [0],
      \"data\": [\"{\\\"input_type\\\": \\\"yaml\\\", \\\"content\\\": \\\"$YAML_CONTENT\\\", \\\"model_version\\\": \\\"boltz1\\\", \\\"output_format\\\": \\\"mmcif\\\", \\\"sampling_steps\\\": 200, \\\"diffusion_samples\\\": 5, \\\"affinity_mw_correction\\\": false}\"]
    }
  }"
```

**Output Processing Example**:
```python
def process_boltz1_output(result, file_path):
    """Process Boltz-1 structure prediction output and extract key metrics."""
    if 'error' in result:
        return {"error": result['error'], "status": result.get('status')}
    
    processed_result = {"file_path": file_path}
    
    if 'predicted_structures' in result:
        structures = result['predicted_structures']
        processed_result.update({
            "num_structures": len(structures),
            "structure_files": [s.get('filename') for s in structures],
            "prediction_successful": True
        })
    
    if 'confidence_scores' in result:
        conf_data = result['confidence_scores'][0]['data']
        processed_result.update({
            "overall_confidence": conf_data.get('overall_confidence'),
            "mean_residue_confidence": np.mean(conf_data.get('per_residue_confidence', [])),
            "confidence_available": True
        })
    
    if 'affinity_scores' in result:
        affinity = result['affinity_scores']
        processed_result.update({
            "binding_affinity": affinity.get('binding_affinity'),
            "affinity_confidence": affinity.get('confidence'),
            "affinity_prediction_available": True
        })
    
    if 'metadata' in result:
        metadata = result['metadata']
        processed_result.update({
            "computation_time": metadata.get('prediction_timestamp'),
            "model_version": metadata.get('model_version'),
            "status": metadata.get('status')
        })
    
    return processed_result
```

## Data and Resource Specification for Deployment

### **Supported Input Formats**

1. **FASTA Format**: Standard protein sequence format for single or multiple proteins
   - Single protein: `>protein_name\nSEQUENCE`
   - Multi-protein: Multiple FASTA entries for protein complexes

2. **YAML Format**: Structured input for complex biomolecular systems
   - Protein-protein complexes with distance constraints
   - Protein-ligand systems with CCD codes or SMILES
   - Multi-chain assemblies with specified interactions

3. **Input Schema Requirements**:
   - `input_type`: "fasta" or "yaml"
   - `content`: Sequence data or YAML structure definition
   - `model_version`: "boltz1"
   - `output_format`: "pdb" or "mmcif" (default: "mmcif")

### **Configurable Parameters**

**Core Sampling Parameters**:
- `sampling_steps`: Number of diffusion sampling steps (default: 200, range: 50-500)
- `diffusion_samples`: Number of independent samples (default: 1, max: 25)
- `recycling_steps`: Number of structure recycling iterations (default: 3, AF3-mode: 10)
- `step_scale`: Diffusion step scaling factor (default: 1.638)

**Advanced Options**:
- `use_msa_server`: Enable external MSA server for enhanced alignments
- `use_potentials`: Apply inference-time structural potentials
- `max_parallel_samples`: Control parallel processing (default: 1)
- `subsample_msa`: Enable MSA subsampling for efficiency
- `num_subsampled_msa`: Number of MSA sequences to retain (default: 1024)

**Affinity Prediction** (for ligand complexes):
- `affinity_mw_correction`: Apply molecular weight correction
- `sampling_steps_affinity`: Dedicated sampling steps for affinity (default: 200)
- `diffusion_samples_affinity`: Multiple samples for affinity estimation (default: 5)

### **Supported Molecular Systems**

1. **Protein Structure Prediction**: Single-chain and multi-domain proteins
2. **Protein-Protein Complexes**: Binary and higher-order assemblies
3. **Protein-Ligand Systems**: 
   - Small molecule ligands via SMILES notation
   - Standard ligands via Chemical Component Dictionary (CCD) codes
   - Binding affinity prediction capabilities
4. **Multi-Chain Assemblies**: Complex biomolecular systems with specified constraints

### **Output Specifications**

- **Structure Files**: PDB or mmCIF format with full atomic coordinates
- **Confidence Scores**: Per-residue and overall confidence metrics
- **Affinity Predictions**: Binding affinity estimates for ligand complexes
- **Metadata**: Comprehensive prediction parameters and timing information

### **Performance Modes**

- **Fast Mode**: Reduced parameters for quick predictions (1 recycle, 50 steps)
- **Standard Mode**: Balanced quality and speed (3 recycles, 200 steps)
- **High Quality Mode**: Enhanced accuracy with potentials (5 recycles, 300 steps)
- **AlphaFold3 Mode**: AF3-comparable parameters (10 recycles, 25 samples)

For detailed technical specifications and advanced usage examples, refer to the [Boltz-1 GitHub repository](https://github.com/jwohlwend/boltz) and technical documentation.

### **License and Availability**

Boltz-1 is released under the MIT License, making it freely available for both academic and commercial use. This open-source approach democratizes access to state-of-the-art biomolecular structure prediction capabilities.