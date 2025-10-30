<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `description.md` is required. -->

Boltz‑1 is an open-source biomolecular structure prediction model for proteins, protein–protein assemblies, and protein–ligand complexes, delivering high-fidelity 3D structural hypotheses to accelerate drug discovery, structural biology, and biotechnology workflows. It emphasizes reproducible, permissionless access under the MIT license, lowering barriers for researchers and organizations that need accurate multi-component structural inference without proprietary constraints. By combining broad complex coverage with efficient inference, Boltz‑1 enables rapid iteration in early discovery pipelines, target validation, and interaction analysis.

The model supports multi-chain assemblies, protein–ligand binding pose prediction, and flexible recycling with diffusion-based sampling to refine structural hypotheses. It integrates MSAs and ligand conformers through a streamlined preprocessing pipeline, and was trained on biologically assembled PDB structures released before 2021-09-30 (resolution-filtered) together with a distillation corpus, using MSAs generated via ColabFold/MMseqs2 and RDKit-derived initial ligand conformers. This data-efficient training regimen was designed to reduce compute while preserving accuracy, leveraging architectural and sampling optimizations. Boltz‑1 is deployable on a single workstation or scalable cloud environments, making it suitable for both exploratory research and production integration.

