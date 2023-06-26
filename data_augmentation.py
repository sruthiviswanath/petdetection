#This is sample python script to perform data augmentation to dataset.
import torchvision.transforms as transforms
from PIL import Image

def apply_data_augmentation(image_path):
   
    # Load the image
    image = Image.open(image_path)

    # Define data augmentation transformations
    augmentation_transforms = transforms.Compose([
        transforms.RandomRotation(30),   # Randomly rotate the image by 30 degrees
        transforms.RandomResizedCrop(224),  # Randomly crop and resize the image to 224x224
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Adjust color values
    ])

    # Apply data augmentation transformations
    augmented_image = augmentation_transforms(image)

    return augmented_image

# Example usage
image_path = 'path/to/image.jpg'
augmented_image = apply_data_augmentation(image_path)

# Save the augmented image
augmented_image.save('path/to/augmented_image.jpg')
