from flask import Flask, request, jsonify,render_template
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torchvision.transforms as tt
import torchvision.models as models

app = Flask(__name__)
import torch

# Load the model on CPU
device = torch.device('cpu')

# Absolute path to the model file
model_path = "C:\\Users\\vinay\\OneDrive\\Desktop\\Prediction\\venv102\\new_model_path.pth"

labels = ['adenocarcinoma','large.cell.carcinoma','normal','squamous.cell.carcinoma']


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}],{} train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, "last_lr: {:.5f},".format(result['lrs'][-1]) if 'lrs' in result else '',
            result['train_loss'], result['val_loss'], result['val_acc']))

class PetsModel(ImageClassificationBase):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        # Use a pretrained model
        self.network = models.vgg16(pretrained=pretrained)
        # Replace last layer
        self.network.classifier= nn.Linear(self.network.classifier[0].in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)
    
model =model = PetsModel(len(labels))

state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Load state dict to the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Use the model for inference

# Define image transformations
transform = tt.Compose([
                        
                         tt.Resize([256,256]),
                         tt.ToTensor(),
                         tt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = transform(img).unsqueeze(0)
    return img
@app.route('/')
def index():
    return render_template('index.html')
# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if 'file' not in request.files:
        return jsonify({'error': 'No image found'})
    with torch.no_grad():
        img =preprocess_image(file)
        output = model(img)
        
        _, preds  = torch.max(output, dim=1)
    # Process output as needed
    return jsonify({'prediction': labels[preds[0].item()]})

if __name__ == '__main__':
    app.run(debug=True)
