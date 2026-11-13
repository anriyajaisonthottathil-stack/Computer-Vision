import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys


# ==========================================
# 1. SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "attention_cnn.pth"

IMAGE_SIZE = 96
NUM_CLASSES = 37

DEVICE = torch.device("cpu")


# ==========================================
# 2. IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# ==========================================
# 3. CNN BLOCK
# ==========================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.block(x)


# ==========================================
# 4. CHANNEL ATTENTION
# ==========================================

class ChannelAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()

        hidden_channels = max(channels // 8, 1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1)
        )

    def forward(self, x):

        average = F.adaptive_avg_pool2d(x, 1)
        maximum = F.adaptive_max_pool2d(x, 1)

        attention = torch.sigmoid(
            self.shared_mlp(average) +
            self.shared_mlp(maximum)
        )

        return x * attention


# ==========================================
# 5. SPATIAL ATTENTION
# ==========================================

class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=7,
            padding=3,
            bias=False
        )

    def forward(self, x):

        average = torch.mean(x, dim=1, keepdim=True)

        maximum = torch.max(
            x,
            dim=1,
            keepdim=True
        )[0]

        combined = torch.cat(
            [average, maximum],
            dim=1
        )

        attention_map = torch.sigmoid(
            self.conv(combined)
        )

        return x * attention_map, attention_map


# ==========================================
# 6. CBAM
# ==========================================

class CBAM(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):

        x = self.channel_attention(x)

        x, attention_map = self.spatial_attention(x)

        return x, attention_map


# ==========================================
# 7. SELF ATTENTION
# ==========================================

class SelfAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.query = nn.Conv2d(
            channels, channels // 8, 1
        )

        self.key = nn.Conv2d(
            channels, channels // 8, 1
        )

        self.value = nn.Conv2d(
            channels, channels, 1
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        batch, channels, height, width = x.size()

        query = self.query(x).view(
            batch, -1, height * width
        ).permute(0, 2, 1)

        key = self.key(x).view(
            batch, -1, height * width
        )

        value = self.value(x).view(
            batch, -1, height * width
        )

        attention = torch.bmm(query, key)

        attention = F.softmax(attention, dim=-1)

        output = torch.bmm(
            value,
            attention.permute(0, 2, 1)
        )

        output = output.view(
            batch, channels, height, width
        )

        return self.gamma * output + x


# ==========================================
# 8. CNN WITH ATTENTION MODEL
# ==========================================

class CNNModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)

        self.cbam = CBAM(128)
        self.self_attention = SelfAttention(128)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x, attention_map = self.cbam(x)

        x = self.self_attention(x)

        x = self.pool(x)

        output = self.classifier(x)

        return output


# ==========================================
# 9. LOAD TRAINED MODEL
# ==========================================

if not MODEL_PATH.exists():

    print("Model file not found!")
    print("Expected location:", MODEL_PATH)
    print("Please run attention.py first.")

    sys.exit(1)


model = CNNModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("Attention CNN model loaded successfully!")


# ==========================================
# 10. SELECT IMAGE
# ==========================================

if len(sys.argv) > 1:

    image_path = Path(sys.argv[1])

else:

    image_path = Path(
        input("Enter the full image path: ").strip().strip('"')
    )


# ==========================================
# 11. PREDICTION
# ==========================================

if not image_path.exists():

    print("Image not found. Check the file path.")

    sys.exit(1)


try:

    image = Image.open(image_path).convert("RGB")

except Exception as error:

    print("Unable to open image:", error)

    sys.exit(1)


image_tensor = transform(image).unsqueeze(0).to(DEVICE)


with torch.no_grad():

    output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    confidence, predicted = torch.max(
        probabilities,
        dim=1
    )


# ==========================================
# 12. DISPLAY PREDICTION
# ==========================================

predicted_class = predicted.item() + 1

print("\n========== CNN PREDICTION ==========")

print("Image:", image_path.name)

print("Predicted class number:", predicted_class)

print("Confidence:", round(confidence.item() * 100, 2), "%")

print("\nTop 3 predictions:")

top_values, top_indices = torch.topk(
    probabilities,
    3,
    dim=1
)

for score, index in zip(
    top_values[0],
    top_indices[0]
):

    print(
        "Class", index.item() + 1,
        "-",
        round(score.item() * 100, 2),
        "%"
    )

print("\n===================================")