import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
from tqdm import trange
from cnn_model import CNNModel
from tqdm import tqdm


def train_cnn(train_imgs, train_lbls, dev_imgs, dev_lbls, batch_size=64, max_epochs=100, patience=5):
    # Prepare datasets
    print("Starting CNN model training...")
    train_imgs = torch.tensor(train_imgs, dtype=torch.float32).unsqueeze(1)
    train_lbls = torch.tensor(train_lbls, dtype=torch.long)
    dev_imgs = torch.tensor(dev_imgs, dtype=torch.float32).unsqueeze(1)
    dev_lbls = torch.tensor(dev_lbls, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(train_imgs, train_lbls), batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(TensorDataset(dev_imgs, dev_lbls), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    # Main training loop with progress bar over epochs
    for epoch in tqdm(range(max_epochs), desc='Training CNN (Epochs)'):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in dev_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(probabilities, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_accuracy = val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Acc = {train_accuracy:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_accuracy:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs("Models", exist_ok=True)
            torch.save(model.state_dict(), "Models/cnn_best_model.pth")
            print("Best model saved at Models/cnn_best_model.pth")
        else:
            epochs_no_improve += 1
            print(f"No validation loss improvement for {epochs_no_improve} epochs")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Plotting
    os.makedirs("Plots/CNN", exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.legend()
    plt.title("Train Loss and Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss / Accuracy")
    plt.savefig("Plots/CNN/Train Loss and Accuracy over Epochs.png")

    plt.figure()
    plt.plot(val_losses, label="Validation Loss")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.legend()
    plt.title("Validation Loss and Accuracy over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss / Accuracy")
    plt.savefig("Plots/CNN/Validation Loss and Accuracy over Epochs.png")

    plt.close('all')
    print("Training plots saved at Plots/CNN/")
    print("Training done!")