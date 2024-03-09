import cv2
import os
import torch
import torchvision.transforms as transforms


# Function to remove watermarks from new images
def remove_watermark(input_image_path, output_folder, model):
    # Load the input image
    image = cv2.imread(input_image_path)
    
    # Convert BGR image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply any necessary preprocessing (resizing, normalization, etc.)
    # For example, you can use torchvision.transforms
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize the image to fit the model input size
        transforms.ToTensor(),
        # Add more preprocessing as needed (e.g., normalization)
    ])
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient computation
    with torch.no_grad():
        # Forward pass
        output_image = model(image)
    
    # Convert the output tensor to numpy array
    output_image = output_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Convert back to BGR for saving using OpenCV
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    
    # Get the filename from the input path
    filename = os.path.basename(input_image_path)
    
    # Save the output image to the output folder with the same name
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, output_image)


##------------ Usage ---------------
    
# Assuming 'model' is your pre-trained watermark removal model
# 'input_folder' is the folder containing images with watermarks
# 'output_folder' is the folder where you want to save the watermark-removed images

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate over each image in the input folder
for image_file in os.listdir(input_folder):
    if image_file.endswith(".jpg") or image_file.endswith(".png"):  # Adjust file extensions as needed
        # Construct the input image path
        input_image_path = os.path.join(input_folder, image_file)
        
        # Remove watermark and save the result
        remove_watermark(input_image_path, output_folder, model)
