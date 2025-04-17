import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
from tqdm import tqdm
from sklearn.metrics import jaccard_score  # We'll use this for IoU computation

# ===================== CONFIGURATION =====================
shape_in = 572
shape_out = 388
shape_orig = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "saved_images2"
os.makedirs(save_dir, exist_ok=True)
log_file = "train_log.csv"

# ===================== DATASET =====================
class UNetDataset(Dataset):
    def __init__(self, img_paths, lab_paths):
        self.img_paths = img_paths
        self.lab_paths = lab_paths

    def mirror_pad(self, img):
        return F.pad(img, (92, 92, 92, 92), mode='reflect')

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("L").resize((shape_out, shape_out))
        lab = Image.open(self.lab_paths[idx]).convert("L").resize((shape_out, shape_out))

        img = np.array(img).astype(np.float32) / 255.0
        lab = np.array(lab).astype(np.float32) / 255.0

        img = torch.from_numpy(img).unsqueeze(0)  # shape: (1, H, W)
        lab = torch.from_numpy(lab).long()         # shape: (H, W)
        img = self.mirror_pad(img)
        return img, lab

    def __len__(self):
        return len(self.img_paths)

# ===================== UNET MODEL =====================
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding='same')

    def forward(self, x):
        return self.conv(self.up(x))

def center_crop(enc_feat, dec_feat):
    _, _, H, W = dec_feat.shape
    _, _, H_enc, W_enc = enc_feat.shape
    top = (H_enc - H) // 2
    left = (W_enc - W) // 2
    return enc_feat[:, :, top:top+H, left:left+W]

class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3),
                nn.ReLU(inplace=True)
            )
        filters = [64, 128, 256, 512, 1024]

        self.enc1 = conv_block(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2)
        self.center = conv_block(filters[3], filters[4])

        self.up4 = UpConv(filters[4], filters[3])
        self.dec4 = conv_block(filters[4], filters[3])
        self.up3 = UpConv(filters[3], filters[2])
        self.dec3 = conv_block(filters[3], filters[2])
        self.up2 = UpConv(filters[2], filters[1])
        self.dec2 = conv_block(filters[2], filters[1])
        self.up1 = UpConv(filters[1], filters[0])
        self.dec1 = conv_block(filters[1], filters[0])

        self.final = nn.Conv2d(filters[0], n_classes, kernel_size=1)
        self._initialize_weights()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        center = self.center(self.pool4(e4))

        d4 = self.up4(center)
        d4 = torch.cat([center_crop(e4, d4), d4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([center_crop(e3, d3), d3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([center_crop(e2, d2), d2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([center_crop(e1, d1), d1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.final:
                    init.uniform_(m.weight)
                else:
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# ===================== LOSS FUNCTIONS =====================
def get_class_weights(label):
    label = label.squeeze(1)
    mask_cell = (label == 1).float()
    mask_back = (label == 0).float()
    cell_frac = mask_cell.mean(dim=[1,2], keepdim=True)
    back_frac = 1 - cell_frac
    cell_weight = 1.0 / (cell_frac + 1e-6)
    back_weight = 1.0 / (back_frac + 1e-6)
    return cell_weight * mask_cell + back_weight * mask_back

def weighted_loss(output, target):
    log_probs = F.cross_entropy(output, target, reduction="none")
    weights = get_class_weights(target.unsqueeze(1))
    return (log_probs * weights).mean()

# ===================== UTILITY FUNCTIONS =====================
def save_images_for_visualization(images, labels, predictions, epoch, batch_idx):
    images = images.cpu()
    labels = labels.cpu()
    predictions = predictions.cpu()
    batch_size = min(4, images.size(0))
    for i in range(batch_size):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(images[i].squeeze(), cmap='gray')
        axes[0].set_title(f"Input (Epoch {epoch+1}, Batch {batch_idx+1})")
        axes[1].imshow(labels[i], cmap='gray')
        axes[1].set_title("Ground Truth")
        axes[2].imshow(predictions[i], cmap='gray')
        axes[2].set_title("Prediction")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        fig_path = os.path.join(save_dir, f"epoch_{epoch+1}_batch_{batch_idx+1}_img_{i+1}.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"Saved visualization to {fig_path}")

# Compute IoU for a batch using sklearn's jaccard_score
def compute_batch_iou(preds, labels, n_classes=2):
    # Flatten predictions and labels
    preds_flat = preds.view(-1).cpu().numpy()
    labels_flat = labels.view(-1).cpu().numpy()
    # Compute macro average IoU over classes
    iou = jaccard_score(labels_flat, preds_flat, average='macro', labels=list(range(n_classes)))
    return iou

# ===================== VALIDATION FUNCTION =====================
def validate(model, val_loader):
    model.eval()
    total_loss = 0
    iou_scores = []
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = weighted_loss(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            iou = compute_batch_iou(preds, labels)
            iou_scores.append(iou)

            # Save some visualizations for the first batch of each validation epoch
            if batch_idx == 0:
                save_images_for_visualization(images, labels, preds, epoch=0, batch_idx=batch_idx)
    avg_loss = total_loss / len(val_loader)
    avg_iou = sum(iou_scores) / len(iou_scores)
    return avg_loss, avg_iou

# ===================== TRAINING FUNCTION (WITH CSV LOGGING) =====================
def train_with_validation(model, train_loader, val_loader, optimizer, epochs=50, accum_steps=4, log_file="train_log.csv"):
    model.train()
    # Prepare log data dictionary; we'll add keys for epoch, train loss, val loss, and val IoU
    log_data = {"epoch": [], "train_loss": [], "val_loss": [], "val_iou": []}

    for epoch in range(epochs):
        total_train_loss = 0
        optimizer.zero_grad()

        with tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}]", unit="batch") as pbar:
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = weighted_loss(outputs, labels) / accum_steps
                loss.backward()
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
                total_train_loss += loss.item() * accum_steps

                # Optionally save visuals every few batches
                if (batch_idx + 1) % 117 == 0:
                    preds = torch.argmax(outputs, dim=1)
                    save_images_for_visualization(images, labels, preds, epoch, batch_idx)
                pbar.set_postfix({"Train_CE_Loss": f"{loss.item() * accum_steps:.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss (CrossEntropy): {avg_train_loss:.4f}")
        # Run validation at the end of each epoch
        val_loss, val_iou = validate(model, val_loader)
        print(f"[Epoch {epoch+1}] Val Loss (CrossEntropy): {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        # Save metrics into log data
        log_data["epoch"].append(epoch+1)
        log_data["train_loss"].append(avg_train_loss)
        log_data["val_loss"].append(val_loss)
        log_data["val_iou"].append(val_iou)

        # Save log data to CSV (overwrite after each epoch)
        pd.DataFrame(log_data).to_csv(log_file, index=False)

        # Make sure to revert to training mode
        model.train()

# ===================== INFERENCE FUNCTION =====================
def infer_and_plot(model, dataset, index=0):
    model.eval()
    with torch.no_grad():
        image, label = dataset[index]
        image = image.unsqueeze(0).to(device)
        outputs = model(image)
        pred = torch.argmax(outputs, dim=1).squeeze().cpu()
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(image.cpu().squeeze(), cmap="gray")
        axs[0].set_title("Input")
        axs[1].imshow(pred, cmap="gray")
        axs[1].set_title("Prediction")
        axs[2].imshow(label, cmap="gray")
        axs[2].set_title("Ground Truth")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

# ===================== MAIN SCRIPT =====================
if __name__ == "__main__":

    train_imgs = sorted([str(p) for p in Path("./augmented/train/imgs").glob("*.png")])
    train_labs = sorted([str(p) for p in Path("./augmented/train/labels").glob("*.png")])
    val_imgs = sorted([str(p) for p in Path("./augmented/test/imgs").glob("*.png")])
    val_labs = sorted([str(p) for p in Path("./augmented/test/labels").glob("*.png")])

    train_set = UNetDataset(train_imgs, train_labs)
    val_set = UNetDataset(val_imgs, val_labs)
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)
    

    model = UNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=2e-5, momentum=0.99)

    train_with_validation(model, train_loader, val_loader, optimizer, epochs=50, accum_steps=4, log_file=log_file)

    torch.save(model.state_dict(), "unet_checkpoint2.pt")
