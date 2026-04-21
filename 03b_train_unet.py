"""
03b_train_unet.py - Treino de U-Net para segmentacao semantica de feicoes supraglaciais

Projeto: LACRIO IC - Extracao de Feicoes Supraglaciais
Alternativa ao SAM: modelo semantico que nao precisa de prompt.

Arquitetura:
  - Encoder: ResNet34 pre-treinado (ImageNet)
  - Decoder: 4 blocos upsampling com skip connections
  - Saida: mascara binaria pixel-a-pixel (512x512)
  - Params treinaveis: ~24M (encoder ~21M + decoder ~3M)

Vantagens sobre SAM:
  - Nao precisa de prompt (ponto/bbox) na inferencia
  - Forward pass unico (vs 64 por tile no SAM)
  - Treino e inferencia identicos (sem gap)
  - ~60x mais rapido na inferencia

VRAM: ~2 GB com batch_size=4 (512x512)

Uso:
    python 03b_train_unet.py                           # Treina lakes
    python 03b_train_unet.py --feature lakes           # Feicao especifica
    python 03b_train_unet.py --epochs 100 --lr 1e-4    # Customizar
    python 03b_train_unet.py --no-augment               # Sem augmentation
    python 03b_train_unet.py --neg-ratio 2.0            # Mais negativos
    python 03b_train_unet.py --pretrained-from sam      # Inicializa encoder do SAM ViT
"""

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import Config

# Reutilizar collect_pairs e split_dataset do treino SAM
import importlib
_sam_train = importlib.import_module("03_finetune_sam")
collect_pairs = _sam_train.collect_pairs
split_dataset = _sam_train.split_dataset


# ============================================================================
# Modelo U-Net com encoder ResNet34
# ============================================================================

class ConvBlock(nn.Module):
    """Bloco convolucional duplo: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetResNet34(nn.Module):
    """U-Net com encoder ResNet34 pre-treinado.

    Encoder: ResNet34 (4 blocos residuais)
    Decoder: 4 blocos upsampling com skip connections
    Saida: 1 canal (sigmoid para probabilidade)
    """

    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision.models as models

        resnet = models.resnet34(
            weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        )

        # Encoder: extrair blocos do ResNet34
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64, /2
        self.pool0 = resnet.maxpool                                        # 64, /4
        self.enc1 = resnet.layer1   # 64, /4
        self.enc2 = resnet.layer2   # 128, /8
        self.enc3 = resnet.layer3   # 256, /16
        self.enc4 = resnet.layer4   # 512, /32

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = ConvBlock(512, 256)   # 256 + 256 skip

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)   # 128 + 128 skip

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)    # 64 + 64 skip

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)    # 64 + 64 skip (enc0)

        # Upsampling final para resolucao original
        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x):
        # Encoder
        e0 = self.enc0(x)    # (B, 64, H/2, W/2)
        p0 = self.pool0(e0)  # (B, 64, H/4, W/4)
        e1 = self.enc1(p0)   # (B, 64, H/4, W/4)
        e2 = self.enc2(e1)   # (B, 128, H/8, W/8)
        e3 = self.enc3(e2)   # (B, 256, H/16, W/16)
        e4 = self.enc4(e3)   # (B, 512, H/32, W/32)

        # Decoder com skip connections
        d4 = self.up4(e4)                              # (B, 256, H/16, W/16)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))     # + skip e3

        d3 = self.up3(d4)                              # (B, 128, H/8, W/8)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))     # + skip e2

        d2 = self.up2(d3)                              # (B, 64, H/4, W/4)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))     # + skip e1

        d1 = self.up1(d2)                              # (B, 64, H/2, W/2)
        d1 = self.dec1(torch.cat([d1, e0], dim=1))     # + skip e0

        d0 = self.up0(d1)                              # (B, 32, H, W)
        out = self.final(d0)                           # (B, 1, H, W)

        return out


# ============================================================================
# Dataset
# ============================================================================

def get_train_transform():
    """Pipeline de augmentacao para treino."""
    import albumentations as A

    return A.Compose([
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101, p=0.5
        ),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(70, 150), p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
    ])


def get_copy_paste_source(tile_paths, mask_paths):
    """Retorna indices de tiles positivos para copy-paste."""
    return [i for i, m in enumerate(mask_paths) if m is not None]


class GlacierDataset(Dataset):
    """Dataset para U-Net. Retorna imagem RGB + mascara binaria."""

    def __init__(self, tile_paths, mask_paths, transform=None,
                 img_size=512, copy_paste=False, copy_paste_prob=0.5):
        self.tile_paths = tile_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.img_size = img_size
        self.copy_paste = copy_paste
        self.copy_paste_prob = copy_paste_prob
        self.positive_indices = get_copy_paste_source(tile_paths, mask_paths)

    def __len__(self):
        return len(self.tile_paths)

    def _apply_copy_paste(self, image, mask):
        """Cola feicao de outro tile positivo."""
        if not self.positive_indices or random.random() > self.copy_paste_prob:
            return image, mask

        src_idx = random.choice(self.positive_indices)
        src_img = cv2.imread(str(self.tile_paths[src_idx]))
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        src_mask = cv2.imread(str(self.mask_paths[src_idx]), cv2.IMREAD_GRAYSCALE)
        src_mask = (src_mask > 127).astype(np.uint8)

        ys, xs = np.where(src_mask > 0)
        if len(xs) == 0:
            return image, mask

        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1
        crop_img = src_img[y1:y2, x1:x2].copy()
        crop_mask = src_mask[y1:y2, x1:x2].copy()

        ch, cw = crop_img.shape[:2]
        h, w = image.shape[:2]
        if ch >= h or cw >= w or ch == 0 or cw == 0:
            return image, mask

        paste_y = random.randint(0, h - ch)
        paste_x = random.randint(0, w - cw)

        alpha = crop_mask.astype(np.float32)
        for c in range(3):
            image[paste_y:paste_y+ch, paste_x:paste_x+cw, c] = (
                image[paste_y:paste_y+ch, paste_x:paste_x+cw, c] * (1 - alpha) +
                crop_img[:, :, c] * alpha
            ).astype(np.uint8)

        mask[paste_y:paste_y+ch, paste_x:paste_x+cw] = np.maximum(
            mask[paste_y:paste_y+ch, paste_x:paste_x+cw],
            crop_mask.astype(np.float32)
        )
        return image, mask

    def __getitem__(self, idx):
        image = cv2.imread(str(self.tile_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mask_paths[idx] is not None:
            mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        if self.copy_paste:
            image, mask = self._apply_copy_paste(image, mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Resize para tamanho fixo
        image = cv2.resize(image, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_NEAREST)

        # Normalizar ImageNet
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return image_tensor, mask_tensor


# ============================================================================
# Loss: BCE + Dice / BCE + Tversky
# ============================================================================

class BCEDiceLoss(nn.Module):
    """Combina BCE com logits + Dice Loss para treino estavel."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (probs * targets).sum()
        dice = 1 - (2 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

        return self.bce_weight * bce + self.dice_weight * dice


class BCETverskyLoss(nn.Module):
    """BCE + Tversky Loss.

    Tversky index: TI = TP / (TP + fp_weight*FP + fn_weight*FN)
    Tversky loss: TL = 1 - TI

    fp_weight > fn_weight → penaliza FP mais que FN → aumenta precisao.
    Recomendado para features finas com oversegmentacao (crevasses).

    Referencia: Salehi et al. (2017) "Tversky loss function for image segmentation"
    """

    def __init__(self, bce_weight=0.3, tversky_weight=0.7,
                 fp_weight=0.7, fn_weight=0.3):
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets)

        probs = torch.sigmoid(logits)
        smooth = 1.0
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        tversky = 1 - (tp + smooth) / (tp + self.fp_weight * fp + self.fn_weight * fn + smooth)

        return self.bce_weight * bce + self.tversky_weight * tversky


# ============================================================================
# Metricas
# ============================================================================

def compute_dice(pred_logits, targets, threshold=0.5):
    """Calcula Dice score para um batch."""
    probs = torch.sigmoid(pred_logits)
    preds = (probs > threshold).float()
    smooth = 1.0
    intersection = (preds * targets).sum()
    return (2 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def compute_metrics_batch(pred_logits, targets, threshold=0.5):
    """Calcula P, R, F1, IoU para um batch."""
    probs = torch.sigmoid(pred_logits)
    preds = (probs > threshold).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou, "dice": compute_dice(pred_logits, targets, threshold).item()}


# ============================================================================
# Treino
# ============================================================================

def train(feature: str, epochs: int = 100, lr: float = 1e-4,
          batch_size: int = 4, img_size: int = 512,
          use_augment: bool = True, neg_ratio: float = 1.0,
          shadow_neg_ratio: float = 0.5, years=None,
          freeze_encoder: bool = False, patience: int = None,
          loss: str = "bce_dice", fp_weight: float = 0.7, fn_weight: float = 0.3):
    """Treina U-Net para uma feicao."""

    print(f"\n{'='*60}")
    print(f"TREINO U-NET: {feature}")
    print(f"{'='*60}")
    print(f"Device: {Config.DEVICE}")
    print(f"Epochs: {epochs} | LR: {lr} | Batch: {batch_size}")
    print(f"Img size: {img_size}x{img_size}")
    print(f"Augmentation: {'SIM' if use_augment else 'NAO'}")
    print(f"Neg ratio: {neg_ratio} | Shadow neg: {shadow_neg_ratio}")
    print(f"Loss: {loss}" + (f" (fp_weight={fp_weight}, fn_weight={fn_weight})" if loss == "bce_tversky" else ""))
    print(f"Anos usados no treino: {years if years else Config.YEARS}")

    # Coletar dados (reutiliza mesma funcao do SAM)
    print(f"\n[1/3] Coletando dados...")
    tile_paths, mask_paths = collect_pairs(
        feature, years=years, neg_ratio=neg_ratio,
        shadow_neg_ratio=shadow_neg_ratio
    )

    if len(tile_paths) == 0:
        print("  Nenhum dado encontrado!")
        return

    # Split treino/validacao
    train_tiles, train_masks, val_tiles, val_masks = split_dataset(
        tile_paths, mask_paths, train_ratio=Config.TRAIN_VAL_SPLIT
    )

    n_train_pos = sum(1 for m in train_masks if m is not None)
    n_val_pos = sum(1 for m in val_masks if m is not None)
    print(f"  Treino: {len(train_tiles)} ({n_train_pos} pos, "
          f"{len(train_tiles) - n_train_pos} neg)")
    print(f"  Val: {len(val_tiles)} ({n_val_pos} pos, "
          f"{len(val_tiles) - n_val_pos} neg)")

    # Datasets
    train_transform = get_train_transform() if use_augment else None
    train_dataset = GlacierDataset(
        train_tiles, train_masks, transform=train_transform,
        img_size=img_size, copy_paste=use_augment, copy_paste_prob=0.5,
    )
    val_dataset = GlacierDataset(
        val_tiles, val_masks, transform=None,
        img_size=img_size, copy_paste=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    # Modelo
    print(f"\n[2/3] Inicializando U-Net (ResNet34 encoder)...")
    model = UNetResNet34(pretrained=True).to(Config.DEVICE)

    if freeze_encoder:
        _encoder_prefixes = ("enc0.", "enc1.", "enc2.", "enc3.", "enc4.", "pool0.")
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in _encoder_prefixes):
                param.requires_grad = False
        n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"  Encoder CONGELADO ({n_frozen:,} params congelados, apenas decoder treinavel)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametros treinaveis: {n_params:,}")

    # Optimizer — discriminative LR: encoder 10x menor que decoder
    _encoder_prefixes = ("enc0.", "enc1.", "enc2.", "enc3.", "enc4.", "pool0.")
    enc_params = [p for n, p in model.named_parameters()
                  if any(n.startswith(pf) for pf in _encoder_prefixes) and p.requires_grad]
    dec_params = [p for n, p in model.named_parameters()
                  if not any(n.startswith(pf) for pf in _encoder_prefixes) and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": lr * 0.1},
        {"params": dec_params, "lr": lr},
    ], weight_decay=1e-4)
    if not freeze_encoder:
        print(f"  Discriminative LR: encoder={lr*0.1:.1e} | decoder={lr:.1e}")

    # Sem warmup quando encoder congelado (decoder ja parte de LR alto)
    warmup_epochs = 0 if freeze_encoder else 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=lr * 0.01
    )

    if loss == "bce_tversky":
        criterion = BCETverskyLoss(bce_weight=0.3, tversky_weight=0.7,
                                   fp_weight=fp_weight, fn_weight=fn_weight)
    else:
        criterion = BCEDiceLoss(bce_weight=0.3, dice_weight=0.7)

    # Treino
    print(f"\n[3/3] Treinando...")
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = patience if patience is not None else Config.EARLY_STOPPING_PATIENCE
    history = {"train_loss": [], "val_dice": [], "val_f1": [], "val_iou": [], "lr": []}

    checkpoint_path = Config.MODELS_DIR / f"unet_{feature}_best.pth"
    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()

    for epoch in range(1, epochs + 1):
        # Warmup LR (aplica ao decoder; encoder mantém 0.1x)
        if epoch <= warmup_epochs:
            warmup_lr = lr * epoch / warmup_epochs
            for i, pg in enumerate(optimizer.param_groups):
                pg["lr"] = warmup_lr * (0.1 if i == 0 else 1.0)
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[-1]["lr"]  # LR do decoder
        history["lr"].append(current_lr)

        # === Treino ===
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"  Epoca {epoch:02d}/{epochs} [treino]")
        for images, masks in pbar:
            images = images.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)

            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss_sum += loss.item() * images.size(0)
                train_count += images.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")

        train_loss = train_loss_sum / max(train_count, 1)
        history["train_loss"].append(train_loss)

        # === Validacao ===
        model.eval()
        all_tp, all_fp, all_fn = 0, 0, 0
        val_dice_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(Config.DEVICE)
                masks = masks.to(Config.DEVICE)

                logits = model(images)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()

                tp = (preds * masks).sum().item()
                fp = (preds * (1 - masks)).sum().item()
                fn = ((1 - preds) * masks).sum().item()

                all_tp += tp
                all_fp += fp
                all_fn += fn

                dice = compute_dice(logits, masks).item()
                val_dice_sum += dice * images.size(0)
                val_count += images.size(0)

        val_dice = val_dice_sum / max(val_count, 1)
        val_precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
        val_recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0.0
        val_iou = all_tp / (all_tp + all_fp + all_fn) if (all_tp + all_fp + all_fn) > 0 else 0.0

        history["val_dice"].append(val_dice)
        history["val_f1"].append(val_f1)
        history["val_iou"].append(val_iou)

        print(f"  Epoca {epoch:02d}: loss={train_loss:.4f} | "
              f"val_dice={val_dice:.4f} | val_f1={val_f1:.4f} | "
              f"val_iou={val_iou:.4f} | P={val_precision:.3f} R={val_recall:.3f}")

        # Salvar melhor modelo (critério: val_f1 micro)
        if val_f1 > best_val_f1 + Config.EARLY_STOPPING_MIN_DELTA:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_dice,
                "val_f1": val_f1,
                "val_iou": val_iou,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "feature": feature,
                "img_size": img_size,
                "architecture": "unet_resnet34",
            }, checkpoint_path)
            print(f"  >>> Melhor modelo salvo! (f1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping na epoca {epoch} "
                      f"(sem melhoria por {patience} epocas)")
                break

    elapsed = time.time() - start

    # Salvar historico
    history_path = Config.MODELS_DIR / f"unet_{feature}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"TREINO CONCLUIDO")
    print(f"{'='*60}")
    print(f"  Melhor epoca: {best_epoch}")
    print(f"  Melhor Val F1: {best_val_f1:.4f}")
    print(f"  Tempo: {elapsed/60:.1f} minutos")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Historico: {history_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Treino de U-Net para segmentacao semantica"
    )
    parser.add_argument("--feature", type=str, default="lakes",
                        choices=list(Config.FEATURES.keys()))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--neg-ratio", type=float, default=1.0,
                        help="Ratio de amostras negativas vs positivas")
    parser.add_argument("--shadow-neg-ratio", type=float, default=0.5,
                        help="Fracao dos negativos que sao hard negatives de sombra")
    parser.add_argument("--years", type=int, nargs="+", default=None,
                        help="Lista de anos a incluir no treino (ex: --years 2016 2017 2018 2019). "
                             "Default: todos os anos em Config.YEARS.")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Congela o encoder ResNet34 e treina apenas o decoder (~3M params). "
                             "Recomendado para datasets pequenos (<100 amostras).")
    parser.add_argument("--patience", type=int, default=None,
                        help="Paciencia do early stopping. Default: Config.EARLY_STOPPING_PATIENCE.")
    parser.add_argument("--loss", type=str, default="bce_dice",
                        choices=["bce_dice", "bce_tversky"],
                        help="Funcao de loss. bce_tversky penaliza FP mais que FN "
                             "(recomendado para crevasses com oversegmentacao).")
    parser.add_argument("--fp-weight", type=float, default=0.7,
                        help="Peso dos falsos positivos no Tversky Loss (default: 0.7). "
                             "Maior = mais conservador = maior precisao.")
    parser.add_argument("--fn-weight", type=float, default=0.3,
                        help="Peso dos falsos negativos no Tversky Loss (default: 0.3).")
    args = parser.parse_args()

    train(
        feature=args.feature,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        img_size=args.img_size,
        use_augment=not args.no_augment,
        neg_ratio=args.neg_ratio,
        shadow_neg_ratio=args.shadow_neg_ratio,
        years=args.years,
        freeze_encoder=args.freeze_encoder,
        patience=args.patience,
        loss=args.loss,
        fp_weight=args.fp_weight,
        fn_weight=args.fn_weight,
    )


if __name__ == "__main__":
    main()
