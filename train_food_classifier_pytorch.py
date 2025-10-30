import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

# Configuration
BASE_DIR = '../FOOD_DATA'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
full_dataset = datasets.ImageFolder(BASE_DIR)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class_names = full_dataset.classes
print(f"Classes: {class_names}")
print(f"Training samples: {train_size}, Validation samples: {val_size}")

# Build CNN model
class FoodCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(FoodCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = FoodCNN(num_classes=len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training loop
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("\nTraining started...")
for epoch in range(EPOCHS):
    model.train()
    train_loss, train_correct = 0.0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
    
    train_loss = train_loss / train_size
    train_acc = train_correct / train_size
    
    # Validation
    model.eval()
    val_loss, val_correct = 0.0, 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
    
    val_loss = val_loss / val_size
    val_acc = val_correct / val_size
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Training Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig('training_history.png')
plt.show()

# Evaluate and confusion matrix
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

final_acc = sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"\nFinal Validation Accuracy: {final_acc*100:.2f}%")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Prediction function
def predict_food(img_path):
    from PIL import Image
    
    nutrition_info = {
        'chicken_curry': {'calories': 250, 'protein': 25, 'fat': 15, 'fiber': 3, 'carbs': 10},
        'chicken_wings': {'calories': 300, 'protein': 27, 'fat': 20, 'fiber': 1, 'carbs': 5},
        'french_fries': {'calories': 312, 'protein': 4, 'fat': 15, 'fiber': 4, 'carbs': 41},
        'grilled_cheese_sandwich': {'calories': 400, 'protein': 15, 'fat': 25, 'fiber': 2, 'carbs': 30},
        'omelette': {'calories': 280, 'protein': 20, 'fat': 22, 'fiber': 0, 'carbs': 2},
        'pizza': {'calories': 285, 'protein': 12, 'fat': 10, 'fiber': 2, 'carbs': 36},
        'samosa': {'calories': 262, 'protein': 6, 'fat': 10, 'fiber': 3, 'carbs': 35}
    }
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = val_transform(img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
    
    predicted_class = class_names[pred_idx.item()]
    confidence = confidence.item() * 100
    nutrition = nutrition_info.get(predicted_class, {})
    
    print(f"\nPredicted: {predicted_class} ({confidence:.2f}% confidence)")
    print("Nutrition Information (per serving):")
    print(f"  Calories: {nutrition.get('calories', 'N/A')} kcal")
    print(f"  Protein: {nutrition.get('protein', 'N/A')}g")
    print(f"  Fat: {nutrition.get('fat', 'N/A')}g")
    print(f"  Fiber: {nutrition.get('fiber', 'N/A')}g")
    print(f"  Carbs: {nutrition.get('carbs', 'N/A')}g")
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_class} ({confidence:.2f}%)")
    plt.tight_layout()
    plt.show()
    
    return {
        'class': predicted_class,
        'confidence': f"{confidence:.2f}%",
        'nutrition': nutrition
    }

# Save model
torch.save(model.state_dict(), 'food_classifier_pytorch.pth')
print("\nModel saved as 'food_classifier_pytorch.pth'")

print("\n" + "="*50)
print("To predict a single image, use:")
print("result = predict_food('path/to/your/image.jpg')")
print("="*50)
