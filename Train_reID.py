import os
import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

# --- Hyperparameters ---
DATA_DIR = 'cattle_reid_dataset'
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
EMBEDDING_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Dataset ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

dataset = ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model ---
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, EMBEDDING_SIZE)
model = model.to(device)

# --- Loss and Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss={running_loss/len(dataloader):.4f}")

# --- Save Model ---
torch.save(model.state_dict(), "cattle_reid.pth")
print("Model saved as cattle_reid.pth")
