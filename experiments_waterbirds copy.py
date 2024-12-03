import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torch import nn, optim
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt
import numpy as np

from tqdm import tqdm

from utils import full_loss, tsne_visualization


transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

trainset = WaterbirdsFullData('waterbird_complete95_forest2water2', 'waterbird_complete95_forest2water2/metadata.csv', 'train', transform = transform)
valset = WaterbirdsFullData('waterbird_complete95_forest2water2', 'waterbird_complete95_forest2water2/metadata.csv', 'val', transform = transform)
testset = WaterbirdsFullData('waterbird_complete95_forest2water2', 'waterbird_complete95_forest2water2/metadata.csv', 'test', transform = transform)


batch_size = 256

trainloader = DataLoader(trainset, batch_size = batch_size, shuffle=True, num_workers = 6)
valloader = DataLoader(valset, batch_size = 2*batch_size, shuffle=False, num_workers = 6)

epochs = 8
model = models.resnet50(ResNet50_Weights.IMAGENET1K_V2) # models.resnet50(ResNet50_Weights.IMAGENET1K_V2)  # nothing in () for random init
model.fc = nn.Identity()
optimizer = optim.Adam(model.parameters(), lr = 0.0005)


train_loss_history = []
val_loss_history = []
device_id = 1

device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

alpha = torch.nn.Parameter(torch.tensor(0.75, requires_grad=True, device=device))
alpha_optimizer = optim.Adam([alpha], lr=0.0001)
alpha_values = []


best_val_loss_pretraining = np.inf
for epoch in range(epochs):

    model.train()
    train_loss = 0
    total_steps = 0
    for i, (images, labels, _) in enumerate(tqdm(trainloader)):
        optimizer.zero_grad()
        alpha_optimizer.zero_grad()

        images, labels = images.to(device), labels.to(device)
        features = model(images)
        loss = full_loss(features, labels, alpha = alpha)
        loss.backward()
        optimizer.step()
        alpha_optimizer.step()
        alpha.data = torch.clamp(alpha.data, min=0., max=1.0)
        alpha_values.append(alpha.item())
        total_steps +=1
        train_loss += loss.item()
    
    train_loss_history.append(train_loss / total_steps)    
    tqdm.write(f'Epoch: {epoch}, Loss: {train_loss / total_steps}')

    model.eval()
    with torch.no_grad():
        val_loss = 0
        total_steps
        for i, (images, labels, _) in enumerate(valloader):
            images, labels = images.to(device), labels.to(device)
            features = model(images)
            loss = full_loss(features, labels, alpha = alpha)
            val_loss += loss.item()
            total_steps += 1    
        val_loss_history.append(val_loss / total_steps)
        if val_loss < best_val_loss_pretraining:
            best_val_loss_pretraining = val_loss
            #best_model_state_dict = model.state_dict()

    tqdm.write(f'Epoch: {epoch}, Val Loss: {val_loss / total_steps}')
    best_model_state_dict = model.state_dict()
print("best val loss obtained: ", best_val_loss_pretraining)

##### LOSS PLOTS #####

model.eval()
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(train_loss_history)
axes[0].set_title('Train Loss')
axes[1].plot(val_loss_history)
axes[1].set_title('Val Loss')
plt.savefig('plots/loss_plot_{alpha}.png'.format(alpha=alpha))

##### LOSS PLOTS #####


#### t-SNE ####

model = models.resnet50()
model.fc = nn.Identity()
model.load_state_dict(best_model_state_dict)
model = model.to(device)

batch_size = 256
valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=4)

embeddings = np.zeros((batch_size * len(valloader), 2048))
labels = np.zeros(batch_size * len(valloader))
stratum = np.zeros(batch_size * len(valloader))

model.eval()
with torch.no_grad():
    for i, (images, label, strata) in enumerate(tqdm(valloader)):

        if len(label) != batch_size:
            continue

        images, label = images.to(device), label.to(device)
        features = model(images)
        embeddings[i * batch_size: (i + 1) * batch_size] = features.cpu().numpy()
        labels[i * batch_size: (i + 1) * batch_size] = label.cpu().numpy()
        stratum[i * batch_size: (i + 1) * batch_size] = strata.cpu().numpy()


tsne_visualization(embeddings, labels, stratum, title="t-SNE_{alpha}".format(alpha=alpha), save_path='plots/tsne_{alpha}.png'.format(alpha=alpha))

#### t-SNE ####


#### Classification ####

model = models.resnet50()
model.fc = nn.Identity()
model.load_state_dict(best_model_state_dict)

model.fc = nn.Sequential(nn.Linear(2048, 2))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)
model = model.to(device)

for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

train_accuracy = []
val_accuracy = []

epochs = 10

best_val_accuracy = 0
for epoch in range(epochs):
    model.train()
    train_loss = 0
    total_steps = 0
    correct = 0
    total = 0
    for i, (images, labels, _) in enumerate(tqdm(trainloader)):
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_steps +=1
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_accuracy.append(correct / total)    
    tqdm.write(f'Epoch: {epoch}, Loss: {train_loss / total_steps}, Accuracy: {correct / total}')

    model.eval()
    with torch.no_grad():
        val_loss = 0
        total_steps = 0
        correct = 0
        total = 0
        for i, (images, labels, strata) in enumerate(valloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            total_steps += 1
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        val_accuracy.append(correct / total)

        if correct / total > best_val_accuracy:
            best_val_accuracy = correct / total
            best_model_state_dict = model.state_dict()
    
    tqdm.write(f'Epoch: {epoch}, Val Loss: {val_loss / total_steps}, Val Accuracy: {correct / total}')


print("best accuracy obtained: ", max(val_accuracy))

#### Classification ####


#### Test worst class performance classif ####
test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

model = models.resnet50()
model.fc = nn.Sequential(nn.Linear(2048, 2))
model.load_state_dict(best_model_state_dict)
model = model.to(device)

model.eval()

group_accuracies = np.zeros((2, 2))
total_0_0, total_0_1, total_1_0, total_1_1 = 0, 0, 0, 0
correct_0_0, correct_0_1, correct_1_0, correct_1_1 = 0, 0, 0, 0
best_test_accuracy = 0
total = 0

with torch.no_grad():
    for i, (images, labels, strata) in enumerate(tqdm(test_loader)):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        for j in range(len(labels)):
            if labels[j] == 0 and strata[j] == 0:
                total_0_0 += 1
                if predicted[j] == labels[j]:
                    correct_0_0 += 1
            elif labels[j] == 0 and strata[j] == 1:
                total_0_1 += 1
                if predicted[j] == labels[j]:
                    correct_0_1 += 1
            elif labels[j] == 1 and strata[j] == 0:
                total_1_0 += 1
                if predicted[j] == labels[j]:
                    correct_1_0 += 1
            else:
                total_1_1 += 1
                if predicted[j] == labels[j]:
                    correct_1_1 += 1
        
        total += labels.size(0)
    test_accuracy = (correct_0_0 + correct_0_1 + correct_1_0 + correct_1_1) / total
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy

group_accuracies[0, 0] = correct_0_0 / total_0_0
group_accuracies[0, 1] = correct_0_1 / total_0_1
group_accuracies[1, 0] = correct_1_0 / total_1_0
group_accuracies[1, 1] = correct_1_1 / total_1_1

print(group_accuracies)
print("best val loss obtained: ", best_val_loss_pretraining)
print("best test accuracy obtained: ", best_test_accuracy)
print("alpha values: ", alpha_values)