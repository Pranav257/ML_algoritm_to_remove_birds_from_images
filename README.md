# Image Manipulation with Stable Diffusion: Advanced Inpainting Technique


Tools Used

Python
StableDiffusionInpaintPipeline
PyTorch
YOLO
SAM (Segment Anything Model)
PIL
NumPy
Matplotlib

Aim
The primary objective of this project is to develop and implement advanced inpainting techniques to remove and replace specific elements in images. The project encompasses multiple phases: segmentation, inpainting to remove squirrels, and inpainting to replace birds with new bird representations, culminating in a comprehensive set of transformed images.
Introduction
This project integrates object detection, segmentation, and inpainting to manipulate images. We use YOLO for object detection, SAM for segmentation, and the Stable Diffusion model from Hugging Face for inpainting. The pipeline detects objects (birds and squirrels), segments them, and replaces or removes them based on provided prompts, creating seamless and visually appealing results.
Dataset
The dataset consists of images featuring birds and squirrels near a birdfeeder.
Birds

Bird Image 1
Bird Image 2
Bird Image 3
Bird Image 4
Bird Image 5

Squirrels

Squirrel Image 1
Squirrel Image 2
Squirrel Image 3
Squirrel Image 4
Squirrel Image 5

Method
Phase 1: Image Segmentation
Segmentation partitions an image into meaningful segments for easier analysis. This phase identifies and isolates birds and squirrels using advanced models.

Object Detection: YOLO (You Only Look Once) detects and localizes objects by dividing the image into a grid and predicting bounding boxes and class probabilities.
Segmentation Mask Generation: SAM (Segment Anything Model) generates precise segmentation masks for detected objects based on prompts like points or boxes.

Implementation:

Tools Used: Python, YOLO, SAM, PIL, NumPy, Matplotlib
Process: Objects (birds and squirrels) are detected and segmented. Masks are generated and saved for further processing.

Phase 2: Object Removal (Removing Squirrels)
Inpainting reconstructs missing or damaged image regions. This phase uses the Stable Diffusion Inpainting Pipeline to remove squirrels, filling masked regions with seamless background content.

Inpainting Techniques: Modern deep learning-based methods, like Stable Diffusion, intelligently fill missing regions using generative models.
Stable Diffusion Inpainting: Utilizes the pre-trained "runwayml/stable-diffusion-inpainting" model for high-resolution inpainting.

Implementation:

Inpainting Pipeline Initialization:def initialize_inpaint_pipeline():
    """
    Initialize the inpainting pipeline using StableDiffusionInpaintPipeline.
    Returns:
        inpaint_pipeline (StableDiffusionInpaintPipeline): Initialized inpainting pipeline.
    """
    inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float32)
    inpaint_pipeline = inpaint_pipeline.to("cpu")
    return inpaint_pipeline


Squirrel Removal: Processes input images and masks, resizes them to (512, 512), removes squirrels using the inpainting pipeline, and saves outputs as "-squirrelsRemoved.jpeg".
Tools Used: Python, StableDiffusionInpaintPipeline, PyTorch, PIL, NumPy, Matplotlib

Phase 3: Object Replacement (Replacing Birds)
This phase replaces birds with new bird representations using generative models, specifically the Stable Diffusion Inpainting Pipeline.

Generative Adversarial Networks (GANs): Comprise a generator for creating images and a discriminator for evaluating authenticity, improving image quality.
Stable Diffusion Models: Use a diffusion process for high-quality image generation, effective for inpainting tasks.

Implementation:

Generative Model: The Stable Diffusion Inpainting Pipeline processes each image-mask pair with the prompt "Bird replaced, high resolution" to generate new bird representations.
Integration: Builds on squirrel removal techniques, combining segmentation and generative modeling.
Tools Used: Python, StableDiffusionInpaintPipeline, PyTorch, PIL, NumPy, Matplotlib
Code Example:# Perform inpainting on each image
for i in range(len(image_paths)):
    image_path = image_paths[i]
    mask_path = mask_paths[i]
    init_image = Image.open(image_path).resize((512, 512))
    mask_image = Image.open(mask_path).resize((512, 512))
    prompt = 'Bird replaced, high resolution'
    inpainted_image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
    output_filename = f"-Birdsreplaced{i+2}.jpeg"
    inpainted_image.save(output_filename)



Additional Implementation: Generating Bird Feeder Images
Generates images of birds and squirrels fighting near a birdfeeder using a text prompt.

Prompt: "bird and squirrel fighting near a birdfeeder"
Implementation:for i in range(5):
    image = pipe(prompt).images[0]
    path = f'./generated-images/generated-image-{i+1}.jpeg'
    path = Path(path)
    if not path.is_file():
        Path('./generated-images').mkdir(parents=True, exist_ok=True)
    plt.imsave(path, np.array(image))



Dependencies

Hugging Face: 
runwayml/stable-diffusion-inpainting
hustvl/yolos-tiny


PyTorch
TensorFlow
NumPy
Matplotlib
PIL

Results
The project produced a gallery of transformed images:

Squirrel Removal: Squirrels were seamlessly removed, with inpainted regions blending naturally into the background.
Bird Replacement: Birds were replaced with new, visually engaging species, maintaining image coherence.
Generated Images: New images of birds and squirrels fighting near a birdfeeder were created and saved in the ./generated-images directory.

Filename Convention:

Squirrel removal: "-squirrelsRemoved.jpeg"
Bird replacement: "-Birdsreplaced{i+2}.jpeg"
Generated images: "generated-image-{i+1}.jpeg"

Visual Results:

Phase 1: Image Segmentation: Actual images, masked images, and segmented outputs.
Phase 2: Object Removal: Successful removal of squirrels with seamless backgrounds.
Phase 3: Object Replacement: Birds replaced with new representations, maintaining visual quality.

Conclusion
This project demonstrates the power of modern computer vision techniques by integrating segmentation, inpainting, and generative modeling. The pipeline successfully removes squirrels and replaces birds, producing cohesive and visually appealing images. The combination of YOLO, SAM, and Stable Diffusion showcases the sophistication of advanced image manipulation, offering insights into the artistry of image transformation.
