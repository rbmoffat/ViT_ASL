import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from vit_model import ASLVisionTransformer, ASL_LABELS  # Updated import

# Define the same transform as used in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLVisionTransformer(num_classes=len(ASL_LABELS))  # Updated class name
model.load_state_dict(torch.load('../outputs/asl_vit_model.pth', map_location=device))
model.to(device)
model.eval()

# Function to predict on a single image
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
    
    _, predicted = torch.max(output, 1)
    return ASL_LABELS[predicted.item()]  # Use ASL_LABELS for prediction

# Test on a few images
test_image_dir = '../outputs/test_images/'  # Adjust this path as needed
test_images = os.listdir(test_image_dir)[20:40]  # Test on first 10 images

plt.figure(figsize=(20, 10))
for i, image_name in enumerate(test_images):
    image_path = os.path.join(test_image_dir, image_name)
    prediction = predict(image_path)
    
    plt.subplot(4, 5, i+1)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(f"Predicted: {prediction}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('../outputs/test_predictions.png')
plt.show()

print("Test completed. Predictions saved in outputs/test_predictions.png")
