import torch
from torch.utils.data import Dataset, random_split, ConcatDataset
import h5py
import re

class H5Dataset(Dataset):
    def __init__(self, file_path, one_hot_encode_Y=True, pra_tau_features_in=True):   
        super(H5Dataset, self).__init__()
        self.file_path = file_path
        with h5py.File(file_path, 'r') as f:
            self.X = torch.tensor(f['X'][:], dtype=torch.float32)
            self.y = torch.tensor(f['y'][:], dtype=torch.float32)
            # One-hot encode labels
            if one_hot_encode_Y:
                _, indices = torch.max(self.y, dim=1)
                one_hot = torch.nn.functional.one_hot(indices, num_classes=self.y.size(1))
                one_hot = one_hot.float()
                self.y = one_hot
        if pra_tau_features_in:
            # Extract PRA and TAU from filename
            pra, tau = self._extract_pra_tau(file_path)

            # Convert PRA to one-hot
            max_pra_value = 5
            pra_one_hot = torch.nn.functional.one_hot(torch.tensor([pra]), num_classes=max_pra_value).float()

            # Normalize TAU
            tau = tau / 60.0

            # Add PRA (one-hot) and TAU as new features
            tau_tensor = torch.tensor([[tau]], dtype=torch.float32)
            pra_tau_features = torch.cat((pra_one_hot, tau_tensor), dim=1)
            pra_tau_features = pra_tau_features.repeat(len(self.X), 1)

            self.X = torch.cat((self.X, pra_tau_features), dim=1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def _extract_pra_tau(self, file_path):
        match = re.search(r'pra(\d+)_tau(\d+)', file_path)
        if match:
            pra = int(match.group(1))
            tau = int(match.group(2))
            return pra, tau
        else:
            raise ValueError(f"Could not extract PRA and TAU from {file_path}")

def load_aggregated_datasets(paths, one_hot_encode_Y=True, pra_tau_features_in=True):
    datasets = [H5Dataset(path,one_hot_encode_Y, pra_tau_features_in) for path in paths]
    return ConcatDataset(datasets)

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=1234):
    torch.manual_seed(seed)
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])

if __name__ == '__main__':
    # Example usage and testing
    file_path = "MyDataset_polar/my_HCAS_rect_TrainingData_v6_pra0_tau00.h5"  # Replace with an actual path
    dataset = H5Dataset(file_path)
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample item: {dataset[0][0].shape}, {dataset[0][1].shape}")

    train_set, val_set, test_set = split_dataset(dataset)
    print(f"Train size: {len(train_set)}, Val size: {len(val_set)}, Test size: {len(test_set)}")