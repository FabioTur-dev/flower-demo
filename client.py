import argparse
import flwr as fl
import torch

from model import (
    LeNet5_MNIST,
    LeNet5_CIFAR,
    load_mnist,
    load_cifar10,
    device,
)

# -----------------------------------------------------------
# Local Training
# -----------------------------------------------------------
def train_one_epoch(model, trainloader):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()


# -----------------------------------------------------------
# Local Evaluation
# -----------------------------------------------------------
def evaluate_local(model, testloader):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += loss_fn(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()

    return loss_sum / len(testloader), correct / len(testloader.dataset)


# -----------------------------------------------------------
# Flower NumPyClient
# -----------------------------------------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        # Restituisce i pesi locali
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def fit(self, parameters, config):
        # Carica pesi globali
        for p, new in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new, device=device)

        # Allena 1 epoca
        train_one_epoch(self.model, self.trainloader)

        # Ritorna pesi aggiornati
        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # Carica pesi globali
        for p, new in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new, device=device)

        # Eval locale
        loss, acc = evaluate_local(self.model, self.testloader)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc)}


# -----------------------------------------------------------
# Client Main
# -----------------------------------------------------------
from flwr.client import start_client

SERVER_ADDR = "127.0.0.1:8099"

if __name__ == "__main__":

    print("========================================")
    print("Connecting Flower Client to:", SERVER_ADDR)
    print("========================================")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist")
    args = parser.parse_args()

    # Selezione dataset e modello
    if args.dataset == "mnist":
        model = LeNet5_MNIST().to(device)
        trainloader, testloader = load_mnist()
    else:
        model = LeNet5_CIFAR().to(device)
        trainloader, testloader = load_cifar10()

    # Avvia client Flower
    start_client(
        server_address=SERVER_ADDR,
        client=FlowerClient(model, trainloader, testloader).to_client(),
    )

