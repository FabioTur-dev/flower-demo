import flwr as fl
import torch
from model import LeNet5, load_mnist

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LeNet5().to(device)
trainloader, testloader = load_mnist()

# -----------------------
# Training
# -----------------------
def train_one_epoch():
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

# -----------------------
# Evaluation
# -----------------------
def test():
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_sum = 0
    correct = 0

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += loss_fn(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()

    return loss_sum / len(testloader), correct / len(testloader.dataset)

# -----------------------
# Flower NumPyClient
# -----------------------
class Client(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in model.parameters()]

    def fit(self, parameters, config):
        # Load new parameters
        for p, new in zip(model.parameters(), parameters):
            p.data = torch.tensor(new, device=device)

        # Local training
        train_one_epoch()

        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # Load parameters
        for p, new in zip(model.parameters(), parameters):
            p.data = torch.tensor(new, device=device)

        loss, acc = test()
        return float(loss), len(testloader.dataset), {"accuracy": float(acc)}


# -----------------------
# Start client
# -----------------------
from flwr.client import start_client

if __name__ == "__main__":
    start_client(
        server_address="127.0.0.1:8080",   # IMPORTANT on Windows
        client=Client().to_client(),
    )

