<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `description.md` is required. -->

MedSAM2 is a breakthrough medical image segmentation foundation model that addresses the critical need for efficient and accurate segmentation in precision medicine. Built upon the Segment Anything Model (SAM) 2.1 architecture, it uniquely bridges the gap between 2D and 3D medical image segmentation, offering a unified solution for both volumetric scans and video data. The model stands out for its ability to reduce manual annotation costs by over 85% while maintaining high accuracy across diverse medical imaging applications.

The model has been extensively trained on over 455,000 3D image-mask pairs and 76,000 annotated video frames, spanning multiple organs, pathologies, and imaging protocols. MedSAM2's versatility has been validated through comprehensive user studies involving 5,000 CT lesions, 3,984 liver MRI lesions, and 251,550 echocardiogram video frames. The model incorporates a novel self-sorting memory bank mechanism for dynamic feature selection, enabling robust performance across different medical imaging contexts. This makes it particularly valuable for clinical workflows in cardiology, oncology, and surgical specialties where precise 3D organ and lesion segmentation is critical but traditionally time-consuming. Integrated with user-friendly interfaces for both local and cloud deployment, MedSAM2 represents a practical solution for supporting efficient, scalable, and high-quality segmentation in both research and healthcare environments.

## Model Architecture
MedSAM2 adopts the SAM2 network architecture with four main components:
- Image encoder (Hierarchical Vision Transformer)
- Memory attention module
- Prompt encoder
- Mask decoder

The model uses SAM2.1-Tiny variant with full-model fine-tuning approach, optimized for performance with fewer parameters.