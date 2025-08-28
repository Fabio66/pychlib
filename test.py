# Fabio Carletti 28/08/2025
# Import useful libraries
import numpy as np

import matplotlib as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import torchvision.datasets as datasets

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.device)

#%%

train_ds = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_ds  = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

BS      = 128
epochs  = 10

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BS, shuffle=True)
test_dl  = torch.utils.data.DataLoader(test_ds, batch_size=BS)

#%% Build the model

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.a1  = nn.functional.relu(self.fc1(hidden_dim))
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.a2  = nn.functional.softmax(self.fc2(output_dim))

    def forward(self,x):
        x = self.fc1(x)
        x = self.a1
        x = self.fc2(x)
        x = self.a2
        return x


#%%
if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)

    loss = nn.CrossEntropyLoss()
    model = FFNN(784, 50, output_dim=10)
    optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)

    print("Inizio Training")
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_Y in train_dl:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = loss(output, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            correct = (predicted == batch_Y.data)

            if (epoch+1) % 10 == 0:
                accuracy = 100. * correct / len(train_ds)
                avg_loss = total_loss / len(train_ds)
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Acc:{accuracy:.2f}%")

#%% Inference example
model.eval()
with torch.no_grad():
    test_input = torch.randn(100, 784)
    prediction = model(test_input)
    predicted_class = torch.argmax(prediction, dim=1)
    confidence = torch.softmax(prediction, dim=1)

    print(f"\nEsempipio di Inferenza:")
    print(f"Input: {test_input.numpy().flatten()[:5]}...")  # Prime 5 features
    print(f"Predizione: Classe {predicted_class.item()}")
    print(f"Confidenza: {confidence.numpy().flatten()}")

# Informazioni sul modello
print(f"\nInformazioni modello:")
print(f"Numero parametri: {sum(p.numel() for p in model.parameters())}")
print(f"Numero parametri trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# Architettura del modello
print(f"\nArchitettura:")
print(model)