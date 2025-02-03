import torch

def standard_accuracy(y_pred, y_true):
    correct_predictions = torch.argmax(y_pred, dim=1) == torch.argmax(y_true, dim=1)
    return correct_predictions.float().mean().item()

def custAcc(y_true, y_pred):
    # Trova l'indice dell'advisory predetto con valore massimo
    maxes_pred = torch.argmax(y_pred, dim=1)
    # Trova l'indice dell'advisory corretto (con valore massimo in y_true)
    inds = torch.argmax(y_true, dim=1)
    # Calcola la differenza tra gli indici (valore assoluto)
    diff = torch.abs(inds - maxes_pred).float()
    # Genera un tensore di "1" della stessa forma di diff
    ones = torch.ones_like(diff)
    # Genera un tensore di "0" della stessa forma di diff
    zeros = torch.zeros_like(diff)
    # Se la differenza è < 0.5, assegna 1 (predizione corretta), altrimenti 0
    l = torch.where(diff < 0.5, ones, zeros)
    # Restituisce la media (accuratezza complessiva)
    return torch.mean(l)

if __name__ == '__main__':
    # Example usage and testing
    y_true = torch.tensor([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.float32)
    y_pred = torch.tensor([[0.1, 0.8, 0.05, 0.025, 0.025], [0.7, 0.1, 0.1, 0.05, 0.05]], dtype=torch.float32)

    std_acc = standard_accuracy(y_pred, y_true)
    cust_acc = custAcc(y_true, y_pred)

    print(f"Standard Accuracy: {std_acc}")
    print(f"Custom Accuracy: {cust_acc}")