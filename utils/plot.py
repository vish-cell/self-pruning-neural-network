import torch
import matplotlib.pyplot as plt
from model.prunable import PrunableLinear


def plot_gates(model, save_path="gate_distribution.png"):
    gates = []

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            g = m.get_gates().detach().cpu().flatten()
            gates.append(g)

    gates = torch.cat(gates).numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(gates, bins=50)
    plt.title("Distribution of Gate Values")
    plt.xlabel("Gate value")
    plt.ylabel("Frequency")
    plt.grid()

    plt.savefig(save_path)
    print(f"Saved {save_path}")


def plot_tradeoff(lambdas, accuracies, sparsities):
    plt.figure(figsize=(6, 4))

    plt.plot(lambdas, accuracies, marker='o', label="Accuracy")
    plt.plot(lambdas, sparsities, marker='o', label="Sparsity")

    plt.xscale("log")
    plt.xlabel("Lambda (log scale)")
    plt.ylabel("Value")
    plt.title("Lambda vs Accuracy & Sparsity")
    plt.legend()
    plt.grid()

    plt.savefig("lambda_tradeoff.png")
    print("Saved lambda_tradeoff.png")