from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, label


data = ...  # Your data (e.g., a list of images or tensors)
labels = ...  # Your labels (e.g., a list of integers)

custom_dataset = CustomDataset(data, labels)
