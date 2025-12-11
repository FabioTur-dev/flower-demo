import pandas as pd
import matplotlib.pyplot as plt

fed = pd.read_csv("results/federated.csv")
central = pd.read_csv("results/centralized.csv")

plt.figure(figsize=(9,5))

plt.plot(fed["round"], fed["accuracy"], marker="o", linewidth=2, label="Federated")
plt.plot(central["epoch"], central["accuracy"], marker="s", linewidth=2, label="Centralized")

plt.xlabel("Round / Epoch")
plt.ylabel("Accuracy")
plt.title("Federated vs Centralized Training")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
