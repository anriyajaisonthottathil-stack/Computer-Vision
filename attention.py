import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from PIL import Image


# ==========================================================
# 1. SETTINGS
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_SIZE = 96
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 0.001

TRAIN_LIMIT = 300
VALIDATION_LIMIT = 60
NUM_CLASSES = 37

DEVICE = torch.device("cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DATASET_ROOT = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATASET_ROOT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

print("=" * 60)
print("CNN IMAGE CLASSIFICATION WITH ATTENTION")
print("=" * 60)
print("Device:", DEVICE)


# ==========================================================
# 2. DATASET
# ==========================================================

class PetClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = image.convert("RGB")
        image = self.transform(image)
        return image, label


print("\nLoading Oxford-IIIT Pet dataset...")

original_dataset = OxfordIIITPet(
    root=DATASET_ROOT,
    split="trainval",
    target_types="category",
    download=True
)

dataset = PetClassificationDataset(original_dataset)

print("Total images:", len(dataset))
print("Number of classes:", NUM_CLASSES)


# ==========================================================
# 3. TRAIN / VALIDATION SPLIT
# ==========================================================

train_size = min(TRAIN_LIMIT, len(dataset) - VALIDATION_LIMIT)
validation_size = min(VALIDATION_LIMIT, len(dataset) - train_size)
remaining_size = len(dataset) - train_size - validation_size

train_dataset, validation_dataset, _ = random_split(
    dataset,
    [train_size, validation_size, remaining_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(validation_dataset))


# ==========================================================
# 4. BASIC CNN BLOCK
# ==========================================================

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


# ==========================================================
# 5. CHANNEL ATTENTION
# ==========================================================

class ChannelAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()

        hidden_channels = max(channels // 8, 1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1)
        )

    def forward(self, x):
        average = F.adaptive_avg_pool2d(x, 1)
        maximum = F.adaptive_max_pool2d(x, 1)

        attention = torch.sigmoid(
            self.shared_mlp(average) +
            self.shared_mlp(maximum)
        )

        return x * attention


# ==========================================================
# 6. SPATIAL ATTENTION
# ==========================================================

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
        maximum = torch.max(x, dim=1, keepdim=True)[0]

        combined = torch.cat([average, maximum], dim=1)

        attention_map = torch.sigmoid(self.conv(combined))

        return x * attention_map, attention_map


# ==========================================================
# 7. CBAM
# ==========================================================

class CBAM(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.channel_attention = ChannelAttention(channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        x, attention_map = self.spatial_attention(x)
        return x, attention_map


# ==========================================================
# 8. SELF-ATTENTION
# ==========================================================

class SelfAttention(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)

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

        output = output.view(batch, channels, height, width)

        return self.gamma * output + x


# ==========================================================
# 9. CNN MODEL
# ==========================================================

class CNNModel(nn.Module):

    def __init__(self, use_attention=False):
        super().__init__()

        self.use_attention = use_attention

        self.conv1 = ConvBlock(3, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)

        if self.use_attention:
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

        self.last_attention_map = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.use_attention:
            x, attention_map = self.cbam(x)
            self.last_attention_map = attention_map
            x = self.self_attention(x)

        x = self.pool(x)
        return self.classifier(x)


# ==========================================================
# 10. EVALUATION
# ==========================================================

def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0.0
    average_loss = total_loss / len(loader) if len(loader) > 0 else 0.0

    return accuracy, average_loss


# ==========================================================
# 11. TRAINING
# ==========================================================

def train_model(model, model_name):
    print("\n" + "=" * 50)
    print("Training:", model_name)
    print("=" * 50)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    train_losses = []
    validation_accuracies = []
    best_accuracy = -1.0

    model_path = os.path.join(MODEL_DIR, model_name + ".pth")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        accuracy, validation_loss = evaluate(
            model,
            validation_loader
        )

        train_losses.append(epoch_loss)
        validation_accuracies.append(accuracy)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Training Loss: {epoch_loss:.4f} "
            f"Validation Accuracy: {accuracy:.4f}"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), model_path)

    print("\nCompleted:", model_name)

    return {
        "model": model,
        "accuracy": best_accuracy,
        "losses": train_losses,
        "accuracies": validation_accuracies
    }


# ==========================================================
# 12. TRAIN NORMAL CNN
# ==========================================================

normal_cnn = CNNModel(use_attention=False)
normal_results = train_model(normal_cnn, "normal_cnn")


# ==========================================================
# 13. TRAIN CNN WITH ATTENTION
# ==========================================================

attention_cnn = CNNModel(use_attention=True)
attention_results = train_model(attention_cnn, "attention_cnn")


# ==========================================================
# 14. MODEL COMPARISON
# ==========================================================

comparison = [
    ["Normal CNN", normal_results["accuracy"]],
    ["CNN with Attention", attention_results["accuracy"]]
]

print("\nMODEL COMPARISON")

for name, accuracy in comparison:
    print(name, ":", round(accuracy * 100, 2), "%")

with open(
    os.path.join(RESULT_DIR, "model_comparison.csv"),
    "w",
    newline=""
) as file:
    writer = csv.writer(file)
    writer.writerow(["Model", "Validation Accuracy"])
    writer.writerows(comparison)


# ==========================================================
# 15. ACCURACY GRAPH
# ==========================================================

plt.figure(figsize=(8, 5))

names = [item[0] for item in comparison]
accuracies = [item[1] * 100 for item in comparison]

plt.bar(names, accuracies)
plt.ylabel("Validation Accuracy (%)")
plt.title("Normal CNN vs CNN with Attention")
plt.ylim(0, 100)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "accuracy_comparison.png")
)
plt.close()


# ==========================================================
# 16. TRAINING LOSS GRAPH
# ==========================================================

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, EPOCHS + 1),
    normal_results["losses"],
    marker="o",
    label="Normal CNN"
)

plt.plot(
    range(1, EPOCHS + 1),
    attention_results["losses"],
    marker="o",
    label="CNN with Attention"
)

plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "training_loss.png")
)
plt.close()


# ==========================================================
# 17. ATTENTION VISUALIZATION
# ==========================================================

attention_model_path = os.path.join(
    MODEL_DIR,
    "attention_cnn.pth"
)

attention_cnn.load_state_dict(
    torch.load(attention_model_path, map_location=DEVICE)
)

attention_cnn.eval()

image, actual_label = validation_dataset[0]

with torch.no_grad():
    output = attention_cnn(image.unsqueeze(0).to(DEVICE))

    predicted_label = torch.argmax(output, dim=1).item()

    attention_map = (
        attention_cnn.last_attention_map
        .squeeze()
        .cpu()
        .numpy()
    )

attention_map = np.array(
    Image.fromarray(attention_map.astype(np.float32)).resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )
)

attention_map = (
    attention_map - attention_map.min()
) / (
    attention_map.max() - attention_map.min() + 1e-8
)


# ==========================================================
# 18. DISPLAY RESULTS
# ==========================================================

image_display = image.permute(1, 2, 0).numpy()

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_display)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(attention_map, cmap="jet")
plt.title("Spatial Attention Map")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image_display)
plt.imshow(attention_map, cmap="jet", alpha=0.5)
plt.title("Attention Overlay")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "attention_visualization.png")
)

plt.show()


# ==========================================================
# 19. FINAL OUTPUT
# ==========================================================

print("\n" + "=" * 50)
print("ATTENTION MECHANISM EXPERIMENT COMPLETED")
print("=" * 50)

print(
    "Normal CNN Accuracy:",
    round(normal_results["accuracy"] * 100, 2),
    "%"
)

print(
    "Attention CNN Accuracy:",
    round(attention_results["accuracy"] * 100, 2),
    "%"
)

print("\nCheck the results folder for graphs and visualizations.")
print("\nProgram completed successfully!")
