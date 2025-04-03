import io
import time

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from safetensors.torch import load_file
from PIL import Image

from models.deepfake_detector_net import DeepfakeDetectorNet


print('Loading Flask...')
app = Flask(__name__)

print('Loading model...')
model = DeepfakeDetectorNet()
model.load_state_dict(load_file('models/deepfake-detector-net.safetensors', device='cpu'))
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    width, height = img.size
    if width > height:
        left = (width - height) / 2
        right = (width + height) / 2
        top = 0
        bottom = height
    else:
        left = 0
        right = width
        top = (height - width) / 2
        bottom = (height + width) / 2
    img = img.crop((left, top, right, bottom))
    input = transform(img).unsqueeze(0)
    
    start_time = time.time()
    with torch.no_grad():
        output = model(input)
        prob = F.sigmoid(output).item()
    inference_time_ms = (time.time() - start_time) * 1000
    
    result = 'Real' if prob > 0.5 else 'Fake'
    real_prob = prob
    fake_prob = 1 - prob
    prob = real_prob if result == 'Real' else fake_prob
    
    return jsonify({
        'prediction': result,
        'real_probability': real_prob,
        'fake_probability': fake_prob,
        'inference_time_ms': inference_time_ms,
    })

if __name__ == '__main__':
    app.run(port=5000, debug=False)