$root = ""

mkdir $root
cd $root

mkdir model, training, api, utils, reports, infra

# requirements
@"
torch
torchvision
fastapi
uvicorn
redis
matplotlib
"@ | Out-File requirements.txt

# ---------------- MODEL ----------------

@"
import torch
import torch.nn as nn

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features)*0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return x @ pruned_weights.t() + self.bias

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)
"@ | Out-File model/prunable.py

@"
import torch.nn as nn
from model.prunable import PrunableLinear

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)

    def sparsity_loss(self):
        loss = 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                loss += m.get_gates().sum()
        return loss

    def sparsity_level(self, t=1e-2):
        total, zero = 0, 0
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                g = m.get_gates()
                total += g.numel()
                zero += (g < t).sum().item()
        return 100 * zero / total
"@ | Out-File model/network.py

# ---------------- TRAIN ----------------

@"
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model.network import Net

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_data():
    transform = transforms.ToTensor()
    train = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)

    return (
        torch.utils.data.DataLoader(train, batch_size=128, shuffle=True),
        torch.utils.data.DataLoader(test, batch_size=256)
    )

def train(lambda_val):
    train_loader, test_loader = get_data()
    model = Net().to(device)

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(5):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = loss_fn(out, y) + lambda_val * model.sparsity_loss()

            opt.zero_grad()
            loss.backward()
            opt.step()

    torch.save(model, "model.pth")
    return model

if __name__ == "__main__":
    train(1e-4)
"@ | Out-File training/train.py

# ---------------- EVALUATE ----------------

@"
import torch
from model.network import Net
import torchvision
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate():
    model = torch.load("model.pth").to(device)
    model.eval()

    test = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(test, batch_size=256)

    correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            p = model(x).argmax(1)
            correct += (p==y).sum().item()
            total += y.size(0)

    print("Accuracy:", 100*correct/total)
    print("Sparsity:", model.sparsity_level())

if __name__ == "__main__":
    evaluate()
"@ | Out-File training/evaluate.py

# ---------------- API ----------------

@"
from fastapi import FastAPI
import torch

app = FastAPI()
model = torch.load("model.pth")
model.eval()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict(x: list):
    x = torch.tensor(x).float().unsqueeze(0)
    out = model(x)
    return {"pred": int(out.argmax().item())}
"@ | Out-File api/app.py

# ---------------- PLOT ----------------

@"
import torch
import matplotlib.pyplot as plt

def plot(model):
    g = []
    for m in model.modules():
        if hasattr(m, 'get_gates'):
            g.append(m.get_gates().detach().cpu().flatten())

    g = torch.cat(g).numpy()
    plt.hist(g, bins=50)
    plt.show()
"@ | Out-File utils/plot.py

# ---------------- DOCKER ----------------

@"
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
"@ | Out-File infra/Dockerfile

@"
version: '3'
services:
  app:
    build: .
    ports:
      - "8000:8000"
"@ | Out-File infra/docker-compose.yml

# ---------------- README ----------------

@"
# Self-Pruning Neural Network

## Run

pip install -r requirements.txt  
python training/train.py  
python training/evaluate.py  

## API
uvicorn api.app:app --reload  

## Docker
docker-compose -f infra/docker-compose.yml up --build
"@ | Out-File README.md

Write-Host "Project created successfully!"