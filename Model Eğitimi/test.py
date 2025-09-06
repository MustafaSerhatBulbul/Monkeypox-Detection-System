# coding: utf-8
import torch
import torchvision
from torchvision import datasets, transforms, models
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import numpy as np

# Cihaz seçimi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test için veri dönüşümleri
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Test veri setini yükle
test_data = datasets.ImageFolder('d:/dataset/test', transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

# Modeli yükle
model = resnet18(weights=None)  
model.fc = nn.Linear(model.fc.in_features, 2)  
model.load_state_dict(torch.load('best_monkeypox_model.pt'))
model = model.to(device)
model.eval()  

# Test metriklerini hesapla
test_loss = 0.0
test_correct = 0
all_preds, all_labels = [], []
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        test_correct += torch.sum(preds == labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_acc = test_correct.double() / len(test_data)
test_loss = test_loss / len(test_loader)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Test Loss: {test_loss:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()