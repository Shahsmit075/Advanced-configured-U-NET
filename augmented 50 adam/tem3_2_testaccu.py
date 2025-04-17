import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import jaccard_score, accuracy_score

# ========== CONFIGURATION ==========
shape_out = 388
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "checkpoint_interrupted.pt"

# ========== DATASET ==========
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

        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        lab = torch.from_numpy(lab).long()        # (H, W)
        img = self.mirror_pad(img)
        return img, lab

    def __len__(self):
        return len(self.img_paths)

# ========== MODEL ==========
import torch.nn as nn
import torch.nn.init as init

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

# ========== METRICS ==========
def compute_batch_iou(preds, labels, n_classes=2):
    preds_flat = preds.view(-1).cpu().numpy()
    labels_flat = labels.view(-1).cpu().numpy()
    return jaccard_score(labels_flat, preds_flat, average='macro', labels=list(range(n_classes)))

def compute_pixel_accuracy(preds, labels):
    return (preds == labels).sum().item() / labels.numel()

def weighted_loss(output, target):
    log_probs = F.cross_entropy(output, target, reduction="none")
    label = target.unsqueeze(1)
    mask_cell = (label == 1).float()
    mask_back = (label == 0).float()
    cell_frac = mask_cell.mean(dim=[1,2,3], keepdim=True)
    back_frac = 1 - cell_frac
    cell_weight = 1.0 / (cell_frac + 1e-6)
    back_weight = 1.0 / (back_frac + 1e-6)
    weights = cell_weight * mask_cell + back_weight * mask_back
    return (log_probs * weights.squeeze(1)).mean()

# ========== MAIN ==========
if __name__ == "__main__":
    val_imgs = sorted([str(p) for p in Path("./augmented/test/imgs").glob("*.png")])
    val_labs = sorted([str(p) for p in Path("./augmented/test/labels").glob("*.png")])

    test_set = UNetDataset(val_imgs, val_labs)
    test_loader = DataLoader(test_set, batch_size=4, shuffle=False)

    model = UNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    total_loss = 0
    total_iou = 0
    total_acc = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = weighted_loss(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            total_iou += compute_batch_iou(preds, labels)
            total_acc += compute_pixel_accuracy(preds, labels)

    n_batches = len(test_loader)
    print(f"\n✅ Test Evaluation Complete:")
    print(f"   - Average Test Loss (CrossEntropy): {total_loss / n_batches:.4f}")
    print(f"   - Average IoU: {total_iou / n_batches:.4f}")
    print(f"   - Average Pixel Accuracy: {total_acc / n_batches:.4f}")
