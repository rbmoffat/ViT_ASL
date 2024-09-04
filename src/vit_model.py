import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

class ASLVisionTransformer(nn.Module):
    def __init__(self, num_classes=29):
        super(ASLVisionTransformer, self).__init__()
        
        # Load pre-trained ViT-Base-Patch16-384 model
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384')
        
        # Replace the classification head
        self.vit.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x):
        return self.vit(x).logits

# Define the labels
ASL_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

def create_asl_vit_model():
    model = ASLVisionTransformer(num_classes=len(ASL_LABELS))
    return model

# Example usage
if __name__ == "__main__":
    # Create the model
    model = create_asl_vit_model()
    
    # Generate a sample input tensor (batch_size, channels, height, width)
    sample_input = torch.randn(1, 3, 384, 384)
    
    # Get the model output
    output = model(sample_input)
    
    print(f"Model output shape: {output.shape}")
    print(f"Number of classes: {len(ASL_LABELS)}")