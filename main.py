import os
import shutil
import random
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Veri seti ana dizini
data_dir = 'C:/Users/bused/Desktop/archive'  # içinde 'real' ve 'fake' klasörleri var

# Bölünmüş datasetin kaydedileceği dizinler
base_output_dir = 'C:/Users/bused/Desktop/split_dataset'
train_dir = os.path.join(base_output_dir, 'train')
test_dir = os.path.join(base_output_dir, 'test')

# Eğitim ve test klasörlerini oluştur
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

def split_dataset(data_dir, train_dir, test_dir, split_ratio=0.8):
    for category in ['real', 'fake']:
        category_path = os.path.join(data_dir, category)
        images = os.listdir(category_path)
        random.shuffle(images)

        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        test_images = images[split_index:]

        train_category_path = os.path.join(train_dir, category)
        test_category_path = os.path.join(test_dir, category)

        os.makedirs(train_category_path, exist_ok=True)
        os.makedirs(test_category_path, exist_ok=True)

        for image in train_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(train_category_path, image))

        for image in test_images:
            shutil.copy(os.path.join(category_path, image), os.path.join(test_category_path, image))

# Eğer veri zaten bölünmüşse tekrar bölme
if not (os.path.exists(os.path.join(train_dir, 'real')) and os.path.exists(os.path.join(test_dir, 'real'))):
    print("Dataset bölünüyor...")
    split_dataset(data_dir, train_dir, test_dir)
else:
    print("Dataset zaten bölünmüş.")

# Veri dönüşümleri (hem train hem test için aynı boyut ve normalizasyon)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 3 kanal için normalize
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Dönüştürme ve DataLoader oluşturuldu.")

# Basit CNN modeli
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 3 kanal input
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 2 sınıf
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Eğitim
print("Model eğitilmeye başlandı.")
for epoch in range(5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")
print("Model eğitimi tamamlandı.")

# Test
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test setindeki doğruluk: {accuracy:.2f}%')

# Modeli kaydet
torch.save(model.state_dict(), 'fake_real_model.pth')
print("Model 'fake_real_model.pth' olarak kaydedildi.")

# Görsel tahmin fonksiyonu (aynı dönüşümlerle)
def predict_image(model, image_path):
    model.eval()
    image = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    classes = ['fake', 'real']
    predicted_class = classes[predicted.item()]
    print(f'Tahmin: {predicted_class} ({image_path})')
    return predicted_class

# Modeli yükle
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('fake_real_model.pth', map_location=device))

# Görsel tahmini yap
image_path = 'C:/Users/bused/Desktop/fake.jpg'
prediction = predict_image(model, image_path)
print(f"Görsel tahmini: {prediction}")
