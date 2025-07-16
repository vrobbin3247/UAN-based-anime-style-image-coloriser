import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# UNet Model Definition (same as training)
class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super(UNet, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.upconv4(b)
        d4 = self.decoder4(torch.cat([d4, e4], dim=1))

        d3 = self.upconv3(d4)
        d3 = self.decoder3(torch.cat([d3, e3], dim=1))

        d2 = self.upconv2(d3)
        d2 = self.decoder2(torch.cat([d2, e2], dim=1))

        d1 = self.upconv1(d2)
        d1 = self.decoder1(torch.cat([d1, e1], dim=1))

        return torch.tanh(self.final_conv(d1))


def load_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load the trained UNet model"""
    model = UNet(in_channels=4, out_channels=3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def preprocess_image(image, target_size=(160, 256)):
    """Preprocess image for model input"""
    if isinstance(image, str):
        image = Image.open(image)

    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize and convert to tensor
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    return transform(image)


def create_input_tensor(sketch_image, scribbles_image):
    """Create 4-channel input tensor from sketch and scribbles"""
    # Preprocess both images
    sketch_tensor = preprocess_image(sketch_image)
    scribbles_tensor = preprocess_image(scribbles_image)

    # Convert sketch to grayscale (1 channel)
    sketch_gray = transforms.Grayscale()(sketch_tensor)

    # Combine: sketch_gray (1 channel) + scribbles (3 channels) = 4 channels
    input_tensor = torch.cat([sketch_gray, scribbles_tensor], dim=0)

    return input_tensor.unsqueeze(0)  # Add batch dimension


def predict_colorization(model, sketch_image, scribbles_image, device='cuda'):
    """Generate colorized image from sketch and scribbles"""
    # Create input tensor
    input_tensor = create_input_tensor(sketch_image, scribbles_image)
    input_tensor = input_tensor.to(device)

    # Generate prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Post-process output
    output = output.squeeze(0)  # Remove batch dimension
    output = torch.clamp(output * 0.5 + 0.5, 0, 1)  # Denormalize and clamp

    # Convert to PIL Image
    output_np = output.cpu().numpy().transpose(1, 2, 0)
    output_pil = Image.fromarray((output_np * 255).astype(np.uint8))

    return output_pil


def visualize_results(sketch_image, scribbles_image, target_image, predicted_image, save_path=None):
    """Visualize all images side by side"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Ensure all images are PIL Images and properly sized
    target_size = (160, 256)

    if isinstance(sketch_image, str):
        sketch_pil = Image.open(sketch_image).convert('RGB')
    else:
        sketch_pil = sketch_image.convert('RGB')
    sketch_pil = sketch_pil.resize(target_size)

    if isinstance(scribbles_image, str):
        scribbles_pil = Image.open(scribbles_image).convert('RGB')
    else:
        scribbles_pil = scribbles_image.convert('RGB')
    scribbles_pil = scribbles_pil.resize(target_size)

    if isinstance(target_image, str):
        target_pil = Image.open(target_image).convert('RGB')
    else:
        target_pil = target_image.convert('RGB')
    target_pil = target_pil.resize(target_size)

    predicted_pil = predicted_image.resize(target_size)

    # Display images
    axes[0].imshow(sketch_pil)
    axes[0].set_title('Sketch Input')
    axes[0].axis('off')

    axes[1].imshow(scribbles_pil)
    axes[1].set_title('Color Scribbles')
    axes[1].axis('off')

    axes[2].imshow(target_pil)
    axes[2].set_title('Ground Truth')
    axes[2].axis('off')

    axes[3].imshow(predicted_pil)
    axes[3].set_title('Model Prediction')
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# Main inference function for your uploaded images
def run_inference_on_uploaded_images():
    """
    Run inference on the uploaded images
    You'll need to update the paths to your actual model and image files
    """

    # Model path - update this to your trained model
    model_path = "weights/unet_epoch_200.pt"  # or latest checkpoint

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the trained model
    print("Loading model...")
    model = load_model(model_path, device)
    print("Model loaded successfully!")

    # For your uploaded images, you would typically do:
    # sketch_image = Image.open("path_to_sketch.png")
    # scribbles_image = Image.open("path_to_scribbles.png")
    # target_image = Image.open("path_to_target.png")

    # Since you uploaded images, they should be accessible via file paths
    # Update these paths based on where your images are saved
    sketch_path = "data/sketch2.jpg"  # Update with actual path
    scribbles_path = "data/scribble2.png"  # Update with actual path
    target_path = "data/color.png"
    print("Running inference...")

    # Generate prediction
    predicted_image = predict_colorization(model, sketch_path, scribbles_path, device)

    # Visualize results
    visualize_results(sketch_path, scribbles_path, target_path, predicted_image,
                      save_path="inference_result.png")

    print("Inference completed! Check the visualization above.")

    return predicted_image


# Alternative function if you have PIL images directly
def run_inference_with_pil_images(sketch_pil, scribbles_pil, target_pil, model_path):
    """
    Run inference when you have PIL images directly
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = load_model(model_path, device)

    # Generate prediction
    predicted_image = predict_colorization(model, sketch_pil, scribbles_pil, device)

    # Visualize results
    visualize_results(sketch_pil, scribbles_pil, target_pil, predicted_image)

    return predicted_image


# Example usage
if __name__ == "__main__":
    # For file-based inference
    predicted = run_inference_on_uploaded_images()

    # If you have PIL images directly, use this instead:
    # predicted = run_inference_with_pil_images(sketch_pil, scribbles_pil, target_pil, model_path)