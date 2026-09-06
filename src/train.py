import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from model import TacticalGestureMLP

# --- Hyperparameters ---
BATCH_SIZE = 32
EPOCHS = 35
LEARNING_RATE = 0.001
SAVE_PATH = 'models/tactical_mlp.pth'

os.makedirs('models', exist_ok=True)

# 1. Load Data
print("[INFO] Loading and splitting landmark dataset...")
train_loader, test_loader, label_to_id, id_to_label = get_dataloaders(batch_size=BATCH_SIZE)
num_classes = len(label_to_id)

print(f"[INFO] Detected {num_classes} Gesture Classes: {label_to_id}")

# 2. Instantiate Model, Loss, Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TacticalGestureMLP(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3. Training Loop
print(f"\n[INFO] Starting training on {device} for {EPOCHS} epochs...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100

    # Validation Phase
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = (val_correct / val_total) * 100

    if epoch % 5 == 0 or epoch == EPOCHS:
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# 4. Save Model Weights & Label Metadata
checkpoint = {
    'model_state_dict': model.state_dict(),
    'num_classes': num_classes,
    'label_to_id': label_to_id,
    'id_to_label': id_to_label
}

torch.save(checkpoint, SAVE_PATH)
print(f"\n[SUCCESS] Model successfully saved to {SAVE_PATH}")