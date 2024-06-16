# import optuna
# from optuna_dashboard import run_server 

# storage = optuna.storages.InMemoryStorage()
# study = optuna.create_study(storage=storage, direction="maximize")
# run_server(storage=storage, port=8082)

import json

# Open the file for reading
with open('runs/lr_tuning/PD_nn.json', 'r') as file:
    trials_data = json.load(file)

# Now, trials_data is a Python object containing the data from the JSON file
# printing the 3 trials with highest "value", 
# Filtering out the trials with None values in the "value" field
filtered_trials_data = [trial for trial in trials_data if trial["value"] is not None]

# Sorting the filtered trials based on the "value" attribute in descending order
top_3_trials_filtered = sorted(filtered_trials_data, key=lambda x: x["value"], reverse=True)[:3]

# Printing the top 3 trials from the filtered data
print(top_3_trials_filtered)
