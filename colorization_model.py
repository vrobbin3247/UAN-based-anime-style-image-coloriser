import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


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


class ColorizationModel:
    def __init__(self, model_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = UNet(in_channels=4, out_channels=3)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform_rgb = transforms.Compose([
            transforms.Resize((160, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        self.to_grayscale = transforms.Grayscale()

    def preprocess(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform_rgb(image)

    def prepare_input(self, sketch, scribble):
        sketch_tensor = self.preprocess(sketch)
        scribble_tensor = self.preprocess(scribble)
        sketch_gray = self.to_grayscale(sketch_tensor)
        input_tensor = torch.cat([sketch_gray, scribble_tensor], dim=0)
        return input_tensor.unsqueeze(0).to(self.device)

    def predict(self, sketch, scribble):
        input_tensor = self.prepare_input(sketch, scribble)
        with torch.no_grad():
            output = self.model(input_tensor)
        output = output.squeeze(0)
        output = torch.clamp(output * 0.5 + 0.5, 0, 1)
        np_img = output.cpu().numpy().transpose(1, 2, 0)
        return Image.fromarray((np_img * 255).astype(np.uint8))

    def visualize(self, sketch, scribble, target, prediction, save_path=None):
        imgs = []
        for img in [sketch, scribble, target, prediction]:
            if isinstance(img, str):
                img = Image.open(img)
            imgs.append(img.resize((160, 256)).convert('RGB'))

        titles = ['Sketch', 'Scribbles', 'Target', 'Prediction']
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, img, title in zip(axes, imgs, titles):
            ax.imshow(img)
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()