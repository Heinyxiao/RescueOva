import os
from torchvision import transforms
from PIL import Image

def preprocess_image(image_path, img_size=(224, 224)):
    """Resize and normalize images for ResNet."""
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image)
