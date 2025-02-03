import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
input_size = 9
hidden_layers = 5
hidden_size = 50
output_size = 5
epochs = 100
batch_size = 128
learning_rate = 0.0005
patience = 10

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

# funzione per valutare il modello e calcolare la confusion matrix, 
# deve essere una confusion matrix 5x5 con le classi da 0 a 4 dove la diagonale 
# rappresenta la percentuale di predizioni corrette e le altre celle rappresentano 
# la percentuale di predizioni errate con la relativa classe predetta 
# ex. confusion_matrix[0][1] = 0.1 indica che il 10% delle predizioni della classe 0 è 
# stato predetto come classe 1. La funzione deve restituire la confusion matrix
def evaluate_model_confusion_matrix(model, dataloader):
    correct_per_class = [0] * 5
    total_per_class = [0] * 5
    confusion_matrix = torch.zeros(5, 5)
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            _, actual = torch.max(y, 1)
            for i in range(len(predicted)):
                confusion_matrix[actual[i]][predicted[i]] += 1
                total_per_class[actual[i]] += 1
                if predicted[i] == actual[i]:
                    correct_per_class[actual[i]] += 1
        # dividi valori confusion matrix per total_per_class per ottenere la percentuale se vicino a 0 metti 0
        for i in range(5):
            for j in range(5):
                confusion_matrix[i][j] = confusion_matrix[i][j] / total_per_class[i]

    return confusion_matrix


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

def compare_nn(dataset_paths, loss_type = 'CE', epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, hidden_size=hidden_size):

    # Models
    model_A = MLP(input_size, hidden_size, output_size, hidden_layers, name=f'my_model_complete_{loss_type}', softmax_output = (loss_type == 'CE')).to(device)
    model_B = MLP(input_size, hidden_size, output_size, hidden_layers, name=f'original_model_complete_{loss_type}', softmax_output = (loss_type == 'CE')).to(device)

    # Optimizers
    optimizer_A = optim.Adam(model_A.parameters(), lr=learning_rate)
    optimizer_B = optim.Adam(model_B.parameters(), lr=learning_rate)

    if loss_type == 'CE':
        # Load aggregated datasets
        dataset_1 = load_aggregated_datasets(dataset_paths[0], one_hot_encode_Y=True, pra_tau_features_in=True)
        dataset_2 = load_aggregated_datasets(dataset_paths[1], one_hot_encode_Y=True, pra_tau_features_in=True)
    elif loss_type == 'MSE':
        dataset_1 = load_aggregated_datasets(dataset_paths[0], one_hot_encode_Y=False, pra_tau_features_in=True)
        dataset_2 = load_aggregated_datasets(dataset_paths[1], one_hot_encode_Y=False, pra_tau_features_in=True)
    
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

    def convert_value(value):
      if isinstance(value, np.float32):
        return float(value)
      else:
        return value

    # Testing per classe con tabella di accuratezza
    table_results_1 = test([model_A,model_B], test_dataloader_1, criterion, accuracy_function, mode="table", num_classes=5)
    table_results_2 = test([model_A,model_B], test_dataloader_2, criterion, accuracy_function, mode="table", num_classes=5)


    # Evaluate confusion matrix
    confusion_matrix_1_A = evaluate_model_confusion_matrix(model_A, test_dataloader_1)
    confusion_matrix_2_A = evaluate_model_confusion_matrix(model_A, test_dataloader_2)
    # Evaluate confusion matrix
    confusion_matrix_1_B = evaluate_model_confusion_matrix(model_B, test_dataloader_1)
    confusion_matrix_2_B = evaluate_model_confusion_matrix(model_B, test_dataloader_2)


    table_results_1_dict = {}
    for k, v in table_results_1.items():
      if isinstance(v, pd.Series):
        table_results_1_dict[k] = {idx: convert_value(value) for idx, value in v.items()}
      else:
        table_results_1_dict[k] = convert_value(v)

    table_results_2_dict = {}
    for k, v in table_results_2.items():
      if isinstance(v, pd.Series):
        table_results_2_dict[k] = {idx: convert_value(value) for idx, value in v.items()}
      else:
        table_results_2_dict[k] = convert_value(v)

    return {
        "model_A_on_dataset_1": {"loss": loss_A_on_1, "accuracy": acc_A_on_1},
        "model_A_on_dataset_2": {"loss": loss_A_on_2, "accuracy": acc_A_on_2},
        "model_B_on_dataset_1": {"loss": loss_B_on_1, "accuracy": acc_B_on_1},
        "model_B_on_dataset_2": {"loss": loss_B_on_2, "accuracy": acc_B_on_2},
        "table_results_dataset_1": table_results_1_dict,
        "table_results_dataset_2": table_results_2_dict,
        "confusion_matrix_dataset_1_A": confusion_matrix_1_A.tolist(),
        "confusion_matrix_dataset_2_A": confusion_matrix_2_A.tolist(),
        "confusion_matrix_dataset_1_B": confusion_matrix_1_B.tolist(),
        "confusion_matrix_dataset_2_B": confusion_matrix_2_B.tolist()
    }

if __name__ == "__main__":

    PRA = ['0', '1', '2', '3', '4']
    TAU = ['00', '05', '10', '15', '20', '30', '40', '60']

    dataset_paths_1 = [f"./MyDataset_polar/my_HCAS_rect_TrainingData_v6_pra{pra}_tau{tau}.h5" for pra in PRA for tau in TAU]
    dataset_paths_2 = [f"./OriginalDataset_polar/HCAS_polar_TrainingData_v6_pra{pra}_tau{tau}.h5" for pra in PRA for tau in TAU]

    all_results = {}

    key = f'complete_data'
    results_ce = compare_nn((dataset_paths_1,dataset_paths_2), loss_type='CE')
    results_mse = compare_nn(((dataset_paths_1,dataset_paths_2)), loss_type='MSE')
    all_results[key] = {'CE': results_ce, 'MSE': results_mse}

    for key, value in all_results.items():
        for loss_type, results in value.items():
            for model_key, model_results in results.items():
                if isinstance(model_results, dict):
                    # Converti eventuali Tensors in valori standard
                    results[model_key] = {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in model_results.items()}


    with open("results_complete.json", "w") as file:
        json.dump(all_results, file, indent=4)