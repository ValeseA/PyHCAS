import torch
from tqdm import tqdm

def train_model_with_early_stopping(model, train_dataloader, val_dataloader, criterion, optimizer, accuracy_function, epochs=10, patience=5, device="cpu"):
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in tqdm(range(epochs), desc=f"Training {model.name}"):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        total_batches = 0

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += accuracy_function(outputs, targets)
            total_batches += 1

        avg_train_loss = epoch_loss / len(train_dataloader)
        avg_train_accuracy = epoch_accuracy / total_batches

        val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, accuracy_function, device)

        if epoch % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Training Loss: {avg_train_loss:.4f} | Training Acc {avg_train_accuracy:.4f} | Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                print(f"Epoch {epoch+1}/{epochs} | Training Loss: {avg_train_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), f"models/{model.name}_best_model.pth")

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
            total_accuracy += accuracy_function(outputs, targets)

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    return avg_loss, avg_accuracy