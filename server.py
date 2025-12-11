import flwr as fl

strategy = fl.server.strategy.FedAvg(
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
)

fl.server.start_server(
    server_address="127.0.0.1:8080",
    strategy=strategy,
    config=fl.server.ServerConfig(num_rounds=5),
)
