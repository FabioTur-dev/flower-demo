import flwr as fl
import csv
import os

SERVER_ADDR = "127.0.0.1:8099"

print("=======================================")
print("Flower Server starting at:", SERVER_ADDR)
print("=======================================")

os.makedirs("results", exist_ok=True)
federated_log = "results/federated.csv"

# crea file CSV con header
with open(federated_log, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["round", "accuracy"])


# --------------------------------------------
# Aggregazione dell'accuracy federata
# --------------------------------------------
def aggregate_accuracy(metrics):
    # metrics: [(num_examples, {"accuracy": value}), ...]
    total_examples = 0
    weighted_acc = 0.0
    for num, m in metrics:
        total_examples += num
        weighted_acc += num * m["accuracy"]
    return weighted_acc / total_examples if total_examples > 0 else 0.0


def evaluate_metrics_aggregation_fn(results):
    # results è lista di tuple (num_examples, metrics)
    acc = aggregate_accuracy(results)

    # Append to CSV
    with open(federated_log, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([evaluate_metrics_aggregation_fn.round_num, acc])

    evaluate_metrics_aggregation_fn.round_num += 1
    return {"accuracy": acc}


evaluate_metrics_aggregation_fn.round_num = 1


# --------------------------------------------
# Strategy
# --------------------------------------------
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
)


# --------------------------------------------
# Run server
# --------------------------------------------
if __name__ == "__main__":
    fl.server.start_server(
        server_address=SERVER_ADDR,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=30),
    )
