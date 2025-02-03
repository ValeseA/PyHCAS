import json
import matplotlib.pyplot as plt
import numpy as np
import os

actions = {0: "COC", 1: "WL", 2: "WR", 3: "SL", 4: "SR"}
datasets = {1: "my_dataset", 2: "original_dataset"}
models = {"A": "my_model", "B": "original_model"}

def generate_confusion_matrix_images(results_file, output_dir="images"):
    """
    Generates and saves confusion matrix images as PNG files, handling the _A and _B suffixes.

    Args:
      results_file (str): Path to the JSON results file.
      output_dir (str, optional): Path to save the images. Defaults to "images".
    """

    with open(results_file, "r") as f:
        results = json.load(f)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for loss_type in ["CE", "MSE"]:
      for dataset_num in ["1", "2"]:
        for model_type in ["A", "B"]:
          key = f"confusion_matrix_dataset_{dataset_num}_{model_type}"
          if key in results["complete_data"][loss_type]:
            matrix = results["complete_data"][loss_type][key]
            matrix = np.array(matrix)
          
            # Plotting the confusion matrix
            plt.figure(figsize=(8, 6))
            plt.imshow(matrix, cmap="viridis", interpolation="nearest")
            plt.title(f"Confusion Matrix - {datasets[int(dataset_num)]} - {models[model_type]} - Loss {loss_type}")
            plt.colorbar()
            tick_marks = np.arange(len(matrix))
            plt.xticks(tick_marks, [f"{actions[i]}" for i in range(len(matrix))])
            plt.yticks(tick_marks, [f"{actions[i]}" for i in range(len(matrix))])

            # Add the values inside the plot, must be formatted as percentage
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                  plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] > 0.5 else "black")

            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            plt.savefig(f"./{output_dir}/{loss_type}_matrix_{datasets[int(dataset_num)]}_{models[model_type]}.png")
            plt.close()

if __name__ == "__main__":
    results_file = "results_complete.json"  # Path to your JSON file
    generate_confusion_matrix_images(results_file)