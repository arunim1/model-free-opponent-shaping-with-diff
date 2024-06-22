import json


def list_json_keys(file_path, item):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)[0]

        if isinstance(data, dict):
            print("Top-level keys in the JSON file:")
            for key in data.keys():
                print(f"- {key}")

            if item in data:
                print(f"The key '{item}' exists in the JSON file.")
                print(f"The value associated with the key '{item}' is:")
                print(data[item])
            else:
                print(f"The key '{item}' does not exist in the JSON file.")
        else:
            print("The JSON file does not contain a dictionary at the top level.")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: '{file_path}' is not a valid JSON file.")


# Usage example
file_path = "./runs/non_mfos_ezc_widelrs_diffabs/out.json"
item = "payoff_mat_p2"

list_json_keys(file_path, item)
