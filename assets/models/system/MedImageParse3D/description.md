<!-- DO NOT CHANGE MARKDOWN HEADERS. IF CHANGED, MODEL CARD MAY BE REJECTED BY A REVIEWER -->

<!-- `description.md` is required. -->

Biomedical image analysis is fundamental for biomedical discovery in cell biology, pathology, radiology, and many other biomedical domains. 3D medical images such as CT and MRI play unique roles in clinical practices. MedImageParse 3D is a foundation model for imaging parsing that can jointly conduct segmentation, detection, and recognition for 3D medical images including CT and MRI. Through joint learning, we can improve accuracy for individual tasks and enable novel applications such as segmenting relevant objects in an image through a text prompt, rather than requiring users to laboriously specify the bounding box for each object. 

MedImageParse 3D was trained on a large dataset comprising triples of image, segmentation mask, and textual description. It takes in 3D medical image volume with a text prompt about the target object type (e.g. pancreas in CT), and outputs the corresponding segmentation mask in 3D volume the same shape as the input image. MedImageParse 3D is also able to identify invalid user inputs describing objects that do not exist in the image. MedImageParse 3D can perform object detection, which aims to locate a specific object of interest, including objects with irregular shapes or of small size.

Traditional segmentation models do segmentation alone, requiring a fully supervised mask during training and typically need either manual bounding boxes or automatic proposals at inference if multiple objects are present. Such model doesn’t inherently know which object to segment unless trained specifically for that class, and it can’t take a text query to switch targets. MedImageParse 3D can segment via text prompts describing the object without needing a user-drawn bounding box. This semantic prompt-based approach lets it parse the image and find relevant objects anywhere in the image.

In summary, MedImageParse shows potential to be a building block for an all-in-one tool for biomedical image analysis by jointly solving segmentation, detection, and recognition. It is broadly applicable to different 3D image modalities through text prompting, which may pave a future path for efficient and accurate image-based biomedical discovery when built upon and integrated into an application.


### Model Architecture
MedImageParse 3D is built upon BiomedParse with the BoltzFormer architecture, optimized for locating small objects in 3D images. Leveraging Boltzmann attention sampling mechanisms, it excels at identifying subtle patterns corresponding to biomedical terminologies, as well as extracting contextually relevant information from dense scientific texts. The model is pre-trained on vast 3D medical image datasets, allowing it to generalize across various biomedical domains with high accuracy.
