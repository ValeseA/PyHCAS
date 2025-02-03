import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers, name, softmax_output = False):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        layers.append(nn.Linear(hidden_size, output_size))
        if softmax_output:
            layers.append(nn.Softmax(dim=1)) # Softmax sull'ultima layer
        self.model = nn.Sequential(*layers)
        self.name = name

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    # Example usage/testing
    input_size = 3
    hidden_size = 50
    output_size = 5
    hidden_layers = 2
    model_name = "TestMLP"

    mlp = MLP(input_size, hidden_size, output_size, hidden_layers, model_name)
    print(mlp) # Print the model architecture
    # create a dummy input
    dummy_input = torch.randn(1, input_size)
    # pass it to the model
    output = mlp(dummy_input)
    print(output.shape)