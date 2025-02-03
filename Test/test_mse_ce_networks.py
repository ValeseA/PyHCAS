import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import h5py
import json
from tqdm import tqdm
import pandas as pd
import numpy as np

# Import custom modules
from common.mlp import MLP
from common.dataset import load_aggregated_datasets, split_dataset
from common.utils import standard_accuracy, custAcc
from losses.custom_mse_loss import custom_MSELoss
from train import train_model_with_early_stopping, evaluate_model

# Configuration
input_size = 3
hidden_layers = 2
hidden_size = 50
output_size = 5
epochs = 50
batch_size = 32
learning_rate = 0.001
patience = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)

# Function for testing per class (table of accuracy per class)
def test_model_table(model, dataloader, num_classes):
    total_per_class = np.zeros(num_classes)
    correct_per_class = np.zeros(num_classes)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            true_labels = torch.argmax(y, 1)

            for i in range(num_classes):
                mask = (true_labels == i)
                total_per_class[i] += mask.sum().item()
                correct_per_class[i] += (predicted[mask] == i).sum().item()

    accuracy_per_class = correct_per_class / total_per_class
    return accuracy_per_class, correct_per_class, total_per_class

# Generate the table with the results
def generate_results_table(models, test_loader):
    columns = ["COC", "WL", "WR", "SL", "SR"]
    num_classes = len(columns)
    results = {}
    for model in models:
        accuracy_per_class, correct_per_class, total_per_class = test_model_table(model, test_loader, num_classes)
        results[model.name] = accuracy_per_class
        results['total_per_class'] = total_per_class
    df = pd.DataFrame(results, index=columns)
    return df

# Evaluate the model and return loss and accuracy
def evaluate_model(model, dataloader, criterion, accuracy_function, device="cpu"):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            batch_accuracy = accuracy_function(outputs, targets)
            total_accuracy += batch_accuracy

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    return avg_loss, avg_accuracy

# Test the model
def test(models, dataloader, criterion, accuracy_function, mode="classic", num_classes=None, columns=None):
    if mode == "classic":
        # Classic testing with loss and average accuracy
        results = {}
        for model in models:
            avg_loss, avg_accuracy = evaluate_model(model, dataloader, criterion, accuracy_function, device)
            results[model.name] = {"Loss": avg_loss, "Accuracy": avg_accuracy}
        return pd.DataFrame(results).T
    elif mode == "table":
        # Testing per class with accuracy table
        if num_classes is None:
            raise ValueError("For mode='table', num_classes must be specified.")
        return generate_results_table(models, dataloader)
    else:
        raise ValueError("Invalid mode. Use 'classic' or 'table'.")

def compare_nn(dataset_paths, key, loss_type = 'CE', epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, hidden_size=hidden_size):

    # Models
    model_A = MLP(input_size, hidden_size, output_size, hidden_layers, name=f'my_model_{key}_{loss_type}', softmax_output = (loss_type == 'CE')).to(device)
    model_B = MLP(input_size, hidden_size, output_size, hidden_layers, name=f'original_model_{key}_{loss_type}', softmax_output = (loss_type == 'CE')).to(device)

    # Optimizers
    optimizer_A = optim.Adam(model_A.parameters(), lr=learning_rate)
    optimizer_B = optim.Adam(model_B.parameters(), lr=learning_rate)

    if loss_type == 'CE':
        # Load aggregated datasets
        dataset_1 = load_aggregated_datasets(dataset_paths[0], one_hot_encode_Y=True, pra_tau_features_in=False)
        dataset_2 = load_aggregated_datasets(dataset_paths[1], one_hot_encode_Y=True, pra_tau_features_in=False)
    elif loss_type == 'MSE':
        dataset_1 = load_aggregated_datasets(dataset_paths[0], one_hot_encode_Y=False, pra_tau_features_in=False)
        dataset_2 = load_aggregated_datasets(dataset_paths[1], one_hot_encode_Y=False, pra_tau_features_in=False)
    # Split datasets into train, validation and test
    train_dataset_1, val_dataset_1, test_dataset_1 = split_dataset(dataset_1)
    train_dataset_2, val_dataset_2, test_dataset_2 = split_dataset(dataset_2)

    # Create DataLoaders for train, validation and test
    train_dataloader_1 = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True)
    val_dataloader_1 = DataLoader(val_dataset_1, batch_size=batch_size, shuffle=False)
    test_dataloader_1 = DataLoader(test_dataset_1, batch_size=batch_size, shuffle=False)

    train_dataloader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True)
    val_dataloader_2 = DataLoader(val_dataset_2, batch_size=batch_size, shuffle=False)
    test_dataloader_2 = DataLoader(test_dataset_2, batch_size=batch_size, shuffle=False)

    # Loss function (CrossEntropy)
    if loss_type == 'CE':
        criterion = nn.CrossEntropyLoss()
        accuracy_function = standard_accuracy
    elif loss_type == 'MSE':
        criterion = custom_MSELoss
        accuracy_function = custAcc
    else:
        raise ValueError("Invalid loss type. Use 'CE' or 'MSE'.")

    # Training with early stopping
    train_model_with_early_stopping(model_A, train_dataloader_1, val_dataloader_1, criterion, optimizer_A, accuracy_function, epochs, patience, device)
    train_model_with_early_stopping(model_B, train_dataloader_2, val_dataloader_2, criterion, optimizer_B, accuracy_function, epochs, patience, device)

    # Testing finale sul test set
    loss_A_on_1, acc_A_on_1 = evaluate_model(model_A, test_dataloader_1, criterion, accuracy_function, device)
    loss_A_on_2, acc_A_on_2 = evaluate_model(model_A, test_dataloader_2, criterion, accuracy_function, device)
    loss_B_on_1, acc_B_on_1 = evaluate_model(model_B, test_dataloader_1, criterion, accuracy_function, device)
    loss_B_on_2, acc_B_on_2 = evaluate_model(model_B, test_dataloader_2, criterion, accuracy_function, device)

    # Testing per classe con tabella di accuratezza
    table_results_1 = test([model_A,model_B], test_dataloader_1, criterion, accuracy_function, mode="table", num_classes=5)
    table_results_2 = test([model_A,model_B], test_dataloader_2, criterion, accuracy_function, mode="table", num_classes=5)

    return {
        "model_A_on_dataset_1": {"loss": loss_A_on_1, "accuracy": acc_A_on_1},
        "model_A_on_dataset_2": {"loss": loss_A_on_2, "accuracy": acc_A_on_2},
        "model_B_on_dataset_1": {"loss": loss_B_on_1, "accuracy": acc_B_on_1},
        "model_B_on_dataset_2": {"loss": loss_B_on_2, "accuracy": acc_B_on_2},
        "table_results_dataset_1": table_results_1.to_dict(),
        "table_results_dataset_2": table_results_2.to_dict()
    }

if __name__ == "__main__":

    PRA = ['0', '1', '2', '3', '4']
    TAU = ['00', '05', '10', '15', '20', '30', '40', '60']

    dataset_paths_1 = [f"./MyDataset_polar/my_HCAS_rect_TrainingData_v6_pra{pra}_tau{tau}.h5" for pra in PRA for tau in TAU]
    dataset_paths_2 = [f"./OriginalDataset_polar/HCAS_polar_TrainingData_v6_pra{pra}_tau{tau}.h5" for pra in PRA for tau in TAU]

    all_results = {}

    #results = {f'data{pra}_{tau}': compare_nn((pra, tau)) for pra in PRA for tau in TAU}
    for pra in PRA:
        for tau in TAU:
            key = f'data{pra}_{tau}'
            dataset_paths = ([f"./MyDataset_polar/my_HCAS_rect_TrainingData_v6_pra{pra}_tau{tau}.h5"], [f"./OriginalDataset_polar/HCAS_polar_TrainingData_v6_pra{pra}_tau{tau}.h5"])
            results_ce = compare_nn(dataset_paths, key=key, loss_type='CE')
            results_mse = compare_nn(dataset_paths, key=key, loss_type='MSE')
            all_results[key] = {'CE': results_ce, 'MSE': results_mse}

    for key, value in all_results.items():
        for loss_type, results in value.items():
            for model_key, model_results in results.items():
                if isinstance(model_results, dict):
                    # Converti eventuali Tensors in valori standard
                    results[model_key] = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in model_results.items()}

    with open("results_networks.json", "w") as file:
        json.dump(all_results, file)