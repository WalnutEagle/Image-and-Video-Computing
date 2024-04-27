import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import fcn_model
import fcn_dataset
from tqdm import tqdm

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model
num_classes = 32
model = fcn_model.FCN8s(num_classes).to(device)

# Define the dataset and dataloader
images_dir_train = "train/"
labels_dir_train = "train_labels/"
class_dict_path = "class_dict.csv"
resolution = (384, 512)
batch_size = 16
num_epochs = 50

camvid_dataset_train = fcn_dataset.CamVidDataset(root='CamVid/', images_dir=images_dir_train, labels_dir=labels_dir_train, class_dict_path=class_dict_path, resolution=resolution, crop=True)
dataloader_train = DataLoader(camvid_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

images_dir_val = "val/"
labels_dir_val = "val_labels/"
camvid_dataset_val = fcn_dataset.CamVidDataset(root='CamVid/', images_dir=images_dir_val, labels_dir=labels_dir_val, class_dict_path=class_dict_path, resolution=resolution, crop=False)
dataloader_val = DataLoader(camvid_dataset_val, batch_size=1, shuffle=False, num_workers=4, drop_last=False)

# Define the loss function and optimizer
def loss_fn(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, labels)
    return loss

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Function definitions for evaluation and visualization remain the same
def eval_model(model, dataloader, device, num_classes, save_pred=False):
    model.eval()
    total_loss = 0
    pixel_accs = []
    ious = []

    if save_pred:
        pred_list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            pixel_acc = fcn_dataset.pixel_accuracy(predicted, labels)
            pixel_accs.append(pixel_acc)

            all_ious = fcn_dataset.compute_iou(predicted, labels, num_classes)
            # Instead of directly appending, filter NaNs here if needed
            all_ious.extend([iou for iou in ious if not np.isnan(iou)])

            # Now, all_ious contains no NaN values
            avg_iou = np.mean(all_ious) if all_ious else float('nan')


            if save_pred:
                pred_list.append(predicted.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    avg_pixel_acc = np.mean(pixel_accs)
    avg_iou = np.mean(ious)

    print(f'Average Loss: {avg_loss:.4f}, Pixel Accuracy: {avg_pixel_acc:.4f}, Mean IoU: {avg_iou:.4f}')

    if save_pred:
        pred_list = np.concatenate(pred_list, axis=0)
        np.save('predictions.npy', pred_list)

    model.train()

def visualize_model(model, dataloader, device):
    log_dir = "vis/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model.eval()
    with torch.no_grad():
        for ind, (images, labels) in enumerate(tqdm(dataloader, desc="Visualizing", leave=False)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            images_vis = fcn_dataset.rev_normalize(images)
            img = images_vis[0].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype('uint8')
            label = labels[0].cpu().numpy()
            pred = predicted[0].cpu().numpy()

            # Visualization code remains unchanged

    model.train()

# Training loop with tqdm
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(dataloader_train, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader_train):.4f}')

    # Evaluate on validation set
    eval_model(model, dataloader_val, device, num_classes)

# Evaluate on test set and visualize
print('='*20)
print('Finished Training. Evaluating on test set...')
eval_model(model, dataloader_val, device, num_classes, save_pred=True)

print('Visualizing predictions on test set...')
visualize_model(model, dataloader_val, device)
