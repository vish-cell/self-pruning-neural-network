import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model.network import Net
from utils.plot import plot_gates, plot_tradeoff

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_data():
    transform = transforms.ToTensor()

    train = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    test = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=256)

    return train_loader, test_loader


def train_model(lambda_val, train_loader, test_loader):
    model = Net().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)
            loss = loss_fn(out, y) + lambda_val * model.sparsity_loss()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # evaluation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    sp = model.sparsity_level()

    return model, acc, sp


if __name__ == "__main__":

    lambdas = [1e-6, 1e-3, 1e-1]

    train_loader, test_loader = get_data()

    acc_list = []
    sp_list = []
    models = []

    print("\nLambda  Accuracy        Sparsity (%)")
    print("-" * 40)

    for l in lambdas:
        print(f"\nRunning lambda = {l}")

        model, acc, sp = train_model(l, train_loader, test_loader)

        # 🔥 SAVE EACH MODEL SEPARATELY
        torch.save(model, f"model_{l}.pth")

        acc_list.append(acc)
        sp_list.append(sp)
        models.append(model)

        print(f"{l:<8.0e}{acc:<15.2f}{sp:<15.2f}")

    
    plot_tradeoff(lambdas, acc_list, sp_list)
    plot_gates(models[-1])   