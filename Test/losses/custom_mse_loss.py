import torch
import torch.nn as nn

def custom_MSELoss(y_pred, y_true, loss_factor=40):
    num_outputs = len(y_pred[0])

    difference = y_true - y_pred
    advisory_true = torch.argmax(y_true, dim=1)
    advisory_onehot = nn.functional.one_hot(advisory_true, num_outputs)
    others_onehot = advisory_onehot - 1

    d_optimal = difference * advisory_onehot
    d_suboptimal = difference * others_onehot

    penalized_optimal_loss = loss_factor * (num_outputs - 1) * (torch.square(d_optimal) + torch.abs(d_optimal))
    optimal_loss = torch.square(d_optimal)

    penalized_suboptimal_loss = loss_factor * (torch.square(d_suboptimal) + torch.abs(d_suboptimal))
    suboptimal_loss = torch.square(d_suboptimal)

    optimal_advisory_loss = torch.where(d_optimal > 0, penalized_optimal_loss, optimal_loss)
    suboptimal_advisory_loss = torch.where(d_suboptimal > 0, penalized_suboptimal_loss, suboptimal_loss)

    mean_loss = torch.mean(optimal_advisory_loss + suboptimal_advisory_loss)
    return mean_loss

if __name__ == '__main__':
    # Example usage and testing
    y_true = torch.tensor([[0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.float32)
    y_pred = torch.tensor([[0.1, 0.8, 0.05, 0.025, 0.025], [0.7, 0.1, 0.1, 0.05, 0.05]], dtype=torch.float32)

    loss = custom_MSELoss(y_pred, y_true)
    print(f"Custom MSE Loss: {loss}")