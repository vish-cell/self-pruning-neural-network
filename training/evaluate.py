import torch
import torchvision
import torchvision.transforms as transforms
from utils.plot import plot_gates

device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate():
    lambdas = [1e-6, 1e-3, 1e-1]

    test = torchvision.datasets.CIFAR10(
        "./data",
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    loader = torch.utils.data.DataLoader(test, batch_size=256)

    print("\nEvaluation Results")
    print("-------------------------")
    print("Lambda   Accuracy   Sparsity")

    for l in lambdas:
        model_path = f"model_{l}.pth"

        model = torch.load(model_path, weights_only=False).to(device)
        model.eval()

        correct, total = 0, 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = 100 * correct / total
        sp = model.sparsity_level()

        print(f"{l:<8.0e}{acc:<10.2f}{sp:<10.2f}")

        # 🔥 plot only for highest lambda (best pruning)
        if l == max(lambdas):
            plot_gates(model)


if __name__ == "__main__":
    evaluate()