import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model architecture (same as training)
import torch.nn as nn

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

# Load model
model = FoodCNN(num_classes=7).to(device)
model.load_state_dict(torch.load('models/food_classifier_pytorch.pth', map_location=device))
model.eval()

# Class names
CLASS_NAMES = ['chicken_curry', 'chicken_wings', 'french_fries', 
               'grilled_cheese_sandwich', 'omelette', 'pizza', 'samosa']

# Image preprocessing
IMG_SIZE = 128
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_food(img_path):
    """Predict food class and show nutrition info"""
    nutrition_info = {
        'chicken_curry': {'calories': 250, 'protein': 25, 'fat': 15, 'fiber': 3, 'carbs': 10},
        'chicken_wings': {'calories': 300, 'protein': 27, 'fat': 20, 'fiber': 1, 'carbs': 5},
        'french_fries': {'calories': 312, 'protein': 4, 'fat': 15, 'fiber': 4, 'carbs': 41},
        'grilled_cheese_sandwich': {'calories': 400, 'protein': 15, 'fat': 25, 'fiber': 2, 'carbs': 30},
        'omelette': {'calories': 280, 'protein': 20, 'fat': 22, 'fiber': 0, 'carbs': 2},
        'pizza': {'calories': 285, 'protein': 12, 'fat': 10, 'fiber': 2, 'carbs': 36},
        'samosa': {'calories': 262, 'protein': 6, 'fat': 10, 'fiber': 3, 'carbs': 35}
    }
    
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_tensor = val_transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
    
    predicted_class = CLASS_NAMES[pred_idx.item()]
    confidence = confidence.item() * 100
    nutrition = nutrition_info.get(predicted_class, {})
    
    # Display results
    print(f"\n{'='*60}")
    print(f"🍽️  Predicted: {predicted_class.upper().replace('_', ' ')}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print(f"{'='*60}")
    print("\n🥗 Nutrition Information (per serving):")
    print(f"  • Calories: {nutrition.get('calories', 'N/A')} kcal")
    print(f"  • Protein:  {nutrition.get('protein', 'N/A')}g")
    print(f"  • Fat:      {nutrition.get('fat', 'N/A')}g")
    print(f"  • Fiber:    {nutrition.get('fiber', 'N/A')}g")
    print(f"  • Carbs:    {nutrition.get('carbs', 'N/A')}g")
    print(f"{'='*60}\n")
    
    # Show all class probabilities
    print("📈 All Class Probabilities:")
    probs_np = probs.cpu().numpy()[0]
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs_np)):
        bar = '█' * int(prob * 50)
        print(f"  {name:25s} {prob*100:5.2f}% {bar}")
    
    # Display image
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"{predicted_class.replace('_', ' ').title()}\n({confidence:.1f}% confidence)", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'prediction_{predicted_class}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        'class': predicted_class,
        'confidence': confidence,
        'nutrition': nutrition,
        'all_probabilities': dict(zip(CLASS_NAMES, probs_np.tolist()))
    }

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("\n" + "="*60)
        print("Usage: python test_prediction.py <image_path>")
        print("="*60)
        print("\nExample:")
        print("  python test_prediction.py ../FOOD_DATA/pizza/image_001.jpg")
        print("  python test_prediction.py test_image.jpg")
        print("\n" + "="*60)
    else:
        img_path = sys.argv[1]
        result = predict_food(img_path)
        print(f"\n✅ Prediction saved as: prediction_{result['class']}.png")
