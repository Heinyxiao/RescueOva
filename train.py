import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import ResNetClassifier

def train(model, dataloader, epochs, device):
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for fine-tuning

    for epoch in range(epochs):
        model.train()
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        
        # Save model checkpoint
        torch.save(model.state_dict(), f"results/model_checkpoints/resnet_epoch{epoch+1}.pth")
