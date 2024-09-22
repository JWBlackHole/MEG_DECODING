import json
import csv
import os

ok = [i for i in range(1, 25) if i not in [3, 12, 16, 20, 21]]

def read_json_file(file_path):
    """Read a single JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def read_json_file(file_path):
    """Read a single JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def parse_json_data(entry, sub):
    """Parse a single JSON entry and return a row for the CSV."""
    return {
        "model": "lda",
        "accuracy": entry["accuracy"],
        "precision": entry["precision"],
        "recall": entry["recall"],
        "f1_score": entry["f1_score"],
        "tp": entry["tp"],
        "fp": entry["fp"],
        "tn": entry["tn"],
        "fn": entry["fn"],
        "sub": sub,
        "task": "0,1,2",
        "session": "0,1",
        "target_label": "voiced_phoneme",
        "tmin": -0.1,
        "tmax": 0.3,
        "decim": 10,
        "low pass": 0.5,
        "high pass": 180,
        "clip_percentile": "no",
        "baseline_correction": "no",
        "time_offset": 0,
        "train_data_balanced": "TRUE",
        "test_data_balanced": "TRUE",
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
    csv_file_path = "../../results/csv/results_organized.csv"
    fieldnames = [
        "model", "accuracy", "precision", "recall", "f1_score", "tp", "fp", "tn", "fn",
        "sub", "task", "session", "target_label", "tmin", "tmax", "decim", "low pass",
        "high pass", "clip_percentile", "baseline_correction", "time_offset",
        "train_data_balanced", "test_data_balanced", "description"
    ]

    j=0
    rows = []
    for i in range(start, end + 1):
        file_path = f"../../results/lda/metrics_LDA_{i}.json"
        if os.path.exists(file_path):
            sub = ok[j]
            print("Reading file: ", file_path, sep='\t',  end=' ')
            print(" |  Subject: ", sub)
            json_data = read_json_file(file_path)
            row = parse_json_data(json_data, sub)
            rows.append(row)
        else:
            print(f"File {file_path} does not exist.")
        j+=1

    write_to_csv(csv_file_path, rows, fieldnames)

if __name__ == "__main__":
    start = 98  # Replace with your start value
    end = 116  # Replace with your end value
    main(start, end)