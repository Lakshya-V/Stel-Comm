import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class TacticalGestureDataset(Dataset):
    def __init__(self, features, labels):
        # Convert pandas DataFrames to PyTorch Tensors
        self.x = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_dataloaders(csv_path='data/raw_csv/tactical_landmarks.csv', batch_size=32, test_size=0.2):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}. Run collect_data.py first!")

    df = pd.read_csv(csv_path)

    # 1. Encode string labels to numeric class IDs
    unique_labels = sorted(df['label'].unique())
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    
    encoded_labels = df['label'].map(label_to_id).values
    features = df.drop(columns=['label']).values

    # 2. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        features, encoded_labels, test_size=test_size, random_state=42, stratify=encoded_labels
    )

    # 3. Create Datasets & Loaders
    train_dataset = TacticalGestureDataset(X_train, y_train)
    test_dataset = TacticalGestureDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, label_to_id, id_to_label