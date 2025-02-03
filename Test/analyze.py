import json
import matplotlib.pyplot as plt
import numpy as np
import os
import re



def calculate_average_results(results_file):
    """Calculates and prints average loss and accuracy from results.

    Args:
      results_file (str): Path to the JSON results file.
    """
    with open(results_file, 'r') as f:
        data = json.load(f)

    aggregated_data = {
        "CE": {"model_A": {"dataset_1": {"loss": 0, "accuracy": 0, "count": 0}, "dataset_2": {"loss": 0, "accuracy": 0, "count": 0}},
               "model_B": {"dataset_1": {"loss": 0, "accuracy": 0, "count": 0}, "dataset_2": {"loss": 0, "accuracy": 0, "count": 0}}},
        "MSE": {"model_A": {"dataset_1": {"loss": 0, "accuracy": 0, "count": 0}, "dataset_2": {"loss": 0, "accuracy": 0, "count": 0}},
                "model_B": {"dataset_1": {"loss": 0, "accuracy": 0, "count": 0}, "dataset_2": {"loss": 0, "accuracy": 0, "count": 0}}}
    }

    pattern_aggregates = {
        "data0": {"model_A": {"loss": 0, "accuracy": 0, "count": 0},
                  "model_B": {"loss": 0, "accuracy": 0, "count": 0}},
        "contains_60": {"model_A": {"loss": 0, "accuracy": 0, "count": 0},
                         "model_B": {"loss": 0, "accuracy": 0, "count": 0}}
    }

    for key, values in data.items():
        if not key.startswith("data"):
            continue
        for loss_type, models in values.items():
            for model_dataset, metrics in models.items():
                try:
                    model = "model_A" if "model_A" in model_dataset else "model_B"
                    if "dataset_1" in model_dataset:
                        dataset = "dataset_1"
                    elif "dataset_2" in model_dataset:
                        dataset = "dataset_2"
                    else:
                        continue
                    aggregated_data[loss_type][model][dataset]["loss"] += metrics["loss"]
                    aggregated_data[loss_type][model][dataset]["accuracy"] += metrics["accuracy"]
                    aggregated_data[loss_type][model][dataset]["count"] += 1
                except:
                   continue

            # Gestione dei pattern
        if re.match(r'^data0', key):  # Pattern che iniziano con "data0"
            for loss_type, models in values.items():
              for model_dataset, metrics in models.items():
                try:
                    model = "model_A" if "model_A" in model_dataset else "model_B"
                    pattern_aggregates["data0"][model]["loss"] += metrics["loss"]
                    pattern_aggregates["data0"][model]["accuracy"] += metrics["accuracy"]
                    pattern_aggregates["data0"][model]["count"] += 1
                except:
                    continue

        if "60" in key:  # Pattern che contengono "60"
            for loss_type, models in values.items():
              for model_dataset, metrics in models.items():
                try:
                    model = "model_A" if "model_A" in model_dataset else "model_B"
                    pattern_aggregates["contains_60"][model]["loss"] += metrics["loss"]
                    pattern_aggregates["contains_60"][model]["accuracy"] += metrics["accuracy"]
                    pattern_aggregates["contains_60"][model]["count"] += 1
                except:
                   continue
                    

    # Calcola le medie regolari
    averages = {}
    for loss_type, models in aggregated_data.items():
        averages[loss_type] = {}
        for model, datasets in models.items():
            averages[loss_type][model] = {}
            for dataset, metrics in datasets.items():
                if metrics["count"] > 0:
                    averages[loss_type][model][dataset] = {
                        "loss": metrics["loss"] / metrics["count"],
                        "accuracy": metrics["accuracy"] / metrics["count"],
                    }
                else:
                    averages[loss_type][model][dataset] = {
                        "loss": 0,
                        "accuracy": 0
                    }

    # Calcola le medie dei pattern
    pattern_averages = {
        pattern: {
            model: {
                "loss": metrics[model]["loss"] / metrics[model]["count"] if metrics[model]["count"] > 0 else 0,
                "accuracy": metrics[model]["accuracy"] / metrics[model]["count"] if metrics[model]["count"] > 0 else 0,
            }
            for model in metrics
        }
        for pattern, metrics in pattern_aggregates.items()
    }
    # Print averages
    print("Average Metrics:\n")
    for loss_type, models in averages.items():
        print(f"  {loss_type}:")
        for model, datasets in models.items():
             for dataset, metrics in datasets.items():
               print(f"    {model} - {dataset}: Loss = {metrics['loss']:.4f}, Accuracy = {metrics['accuracy']:.4f}")
    
    # Print pattern averages
    print("\nPattern-Based Averages:\n")
    for pattern, metrics in pattern_averages.items():
        print(f"  {pattern}:")
        for model, values in metrics.items():
            print(f"    {model}: Loss = {values['loss']:.4f}, Accuracy = {values['accuracy']:.4f}")



if __name__ == "__main__":
    results_file = "results_complete.json"  # Path to your JSON file
    results_networks_file = "results_networks.json"  # Path to your JSON file

    calculate_average_results(results_networks_file)