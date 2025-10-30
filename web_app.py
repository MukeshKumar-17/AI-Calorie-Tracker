from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64

app = Flask(__name__)

# Model
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
        return self.classifier(self.features(x))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FoodCNN(num_classes=7).to(device)
model.load_state_dict(torch.load('models/food_classifier_pytorch.pth', map_location=device))
model.eval()

CLASS_NAMES = ['chicken_curry', 'chicken_wings', 'french_fries', 
               'grilled_cheese_sandwich', 'omelette', 'pizza', 'samosa']

NUTRITION = {
    'chicken_curry': {'calories': 250, 'protein': 25, 'fat': 15, 'fiber': 3, 'carbs': 10},
    'chicken_wings': {'calories': 300, 'protein': 27, 'fat': 20, 'fiber': 1, 'carbs': 5},
    'french_fries': {'calories': 312, 'protein': 4, 'fat': 15, 'fiber': 4, 'carbs': 41},
    'grilled_cheese_sandwich': {'calories': 400, 'protein': 15, 'fat': 25, 'fiber': 2, 'carbs': 30},
    'omelette': {'calories': 280, 'protein': 20, 'fat': 22, 'fiber': 0, 'carbs': 2},
    'pizza': {'calories': 285, 'protein': 12, 'fat': 10, 'fiber': 2, 'carbs': 36},
    'samosa': {'calories': 262, 'protein': 6, 'fat': 10, 'fiber': 3, 'carbs': 35}
}

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
    
    predicted_class = CLASS_NAMES[pred_idx.item()]
    confidence_pct = confidence.item() * 100
    nutrition = NUTRITION[predicted_class]
    
    probs_list = probs.cpu().numpy()[0].tolist()
    
    return jsonify({
        'class': predicted_class.replace('_', ' ').title(),
        'confidence': round(confidence_pct, 2),
        'nutrition': nutrition,
        'probabilities': {name.replace('_', ' ').title(): round(prob*100, 2) 
                          for name, prob in zip(CLASS_NAMES, probs_list)}
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
