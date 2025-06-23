import lightning as L
from torch.utils.data import Dataset, DataLoader

# Single torch dataset with transforms
class TorchDataset(Dataset):
    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.data)

# LightningDataModule implementation
class ImageClassificationModule(L.LightningDataModule):
    def __init__(self, data_dir, train_transforms, val_transforms, data_splits=(0.8, 0.15, 0.15), random_seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.train_transform = train_transforms
        self.val_transforms = val_transforms
        self.data_splits = data_splits
        self.random_seed = random_seed

        self.all_data = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self._num_classes = None
        self._class_weights = None

    @property
    def num_classes(self):
		    # Ensure data is prepared if not already
        if self._num_classes is None:
            self.prepare_data()  # Ensure data is downloaded and prepared
            self._num_classes = infer_num_classes(self.data_dir)  # Analyse data and infer number of classes
        return self._num_classes

    @property
    def class_weights(self):
		    # Ensure data is prepared if not already
        if self._class_weights is None:
            self.prepare_data()  # Ensure data is downloaded and prepared
            self._class_weights = calculate_class_weigths(self.data_dir) # Analyse data and calculate class weigths
        return self._class_weights

    def prepare_data(self):
        # Download and unzip data if it is not yet on the machine
        ...

    def setup(self, stage: str = None):
        # Create train-val-test split with our custom function
        self.all_data = load_all_data(self.data_dir)
        train_data, val_data, test_data = random_split_data(self.all_data, self.data_splits, self.random_seed)

        if stage == "fit" or stage is None:
            self.train_dataset = TorchDataset(train_data, self.train_transforms)
            self.val_dataset = TorchDataset(val_data, self.val_transforms)

        if stage == "test" or stage is None:
            self.test_dataset = TorchDataset(test_data, self.val_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)
