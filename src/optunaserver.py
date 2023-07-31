import optuna
from optuna_dashboard import run_server 

storage = optuna.storages.InMemoryStorage()
study = optuna.create_study(storage=storage, direction="maximize")
run_server(storage=storage, port=8082)