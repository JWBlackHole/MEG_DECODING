import json
import csv
import os

ok = [i for i in range(1, 25) if i not in [3, 12, 16, 20, 21]]

def read_json_file(file_path):
    """Read a single JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def parse_json_data(entry):
    """Parse a single JSON entry and return a row for the CSV."""
    description = json.loads(entry["description"])
    sub_ses_task_list = description.get("sub_ses_task_list", [[None]])

    sub = sub_ses_task_list[0][0] if sub_ses_task_list and sub_ses_task_list[0] else None

    return {
        "model": "pca",
        "accuracy": entry["accuracy"],
        "precision": entry["precision"],
        "recall": entry["recall"],
        "f1_score": entry["f1_score"],
        "tp": entry["tp"],
        "fp": entry["fp"],
        "tn": entry["tn"],
        "fn": entry["fn"],
        "sub": sub,
        "task": "1",
        "session": "0",
        "target_label": description.get("target_label", None),
        "pca_model" :  description.get("pca_model", None),
        "tmin": description.get("meg_tmin", None),
        "tmax": description.get("meg_tmax", None),
        "decim": description.get("meg_decim", None),
        "low pass": description.get("preprocess_low_pass", None),
        "high pass": description.get("preprocess_high_pass", None),
        "clip_percentile": description.get("clip_percentile", None),
        "baseline_correction": "no",
        "time_offset": 0,
        "train_data_balanced": "NONE",
        "test_data_balanced": "NONE",
        "description": entry["description"]
    }

def write_to_csv(file_path, rows, fieldnames):
    """Write rows to a CSV file, appending to the existing file."""
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerows(rows)

def main(start, end):
    csv_file_path = "../../results/pca_results.csv"
    fieldnames = [
        "model", "accuracy", "precision", "recall", "f1_score", "tp", "fp", "tn", "fn",
        "sub", "task", "session", "target_label", "pca_model", "tmin", "tmax", "decim", "low pass",
        "high pass", "clip_percentile", "baseline_correction", "time_offset",
        "train_data_balanced", "test_data_balanced", "description"
    ]

    j=0
    rows = []
    for i in range(start, end + 1):
        file_path = f"../../results/metrics_pca_{i}.json"
        if os.path.exists(file_path):
            print("Reading file: ", file_path)
            json_data = read_json_file(file_path)
            row = parse_json_data(json_data)
            rows.append(row)
        else:
            print(f"File {file_path} does not exist.")
        j+=1

    write_to_csv(csv_file_path, rows, fieldnames)

if __name__ == "__main__":
    start = 29  # Replace with your start value
    end = 34  # Replace with your end value
    main(start, end)
    print("done!")