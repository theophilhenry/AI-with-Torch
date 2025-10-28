import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

# Load model
class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, num_classes)
  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    return out

input_size = 784 # 28x28
hidden_size = 100
num_classes = 10
model = NeuralNet(input_size, hidden_size, num_classes)

model.load_state_dict(torch.load('mnist_ffn.pth'))
model.eval()

# Image -> Tensor
def transform_image(image_bytes):
  # We also need to grayscale and 28x28 the image
  transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
  ])
  
  image = Image.open(io.BytesIO(image_bytes))
  return transform(image).unsqueeze(0)

# Predict
def get_prediction(image_tensor):
  image = image_tensor.reshape(-1, 28*28)
  raw_prediction = model(image)
  _, prediction = torch.max(raw_prediction.data, 1)
  return prediction
  
