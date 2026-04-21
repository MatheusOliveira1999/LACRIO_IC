"""
03a_pretrain_satellite.py - Pre-treino do decoder SAM com dados de satelite

Projeto: LACRIO IC - Extracao de Feicoes Supraglaciais
Estrategia: Warm-start do decoder com lagos supraglaciais de satelite (Groelandia)
antes do fine-tuning com dados de drone do Schiaparelli.

Datasets suportados:
  1. SIGSPATIAL Cup 2023 (iHARP/UMBC) - lagos supraglaciais Groelandia
     - Tiles 1024x1024 com mascaras de lagos
     - Satelite ~3m resolucao
     - Download: https://github.com/knowledge-computing/sigspatial-cup-2023

  2. NASA-IMPACT - lagos supraglaciais PlanetScope
     - Tiles 96x96, ~3m resolucao
     - Download: https://github.com/NASA-IMPACT/veda-ai-supraglacial_segmentation

  3. Custom - qualquer dataset com estrutura tiles/ + masks/

Uso:
    # Preparar dataset (ver instrucoes abaixo)
    # Pre-treinar decoder com dados de satelite
    python 03a_pretrain_satellite.py --data-dir data/satellite_lakes --epochs 20

    # Depois, fine-tunar com dados de drone (usa checkpoint do pre-treino)
    python 03_finetune_sam.py --feature lakes --pretrained models/pretrained_satellite_lakes.pth

Estrutura esperada do dataset:
    data/satellite_lakes/
    ├── tiles/          # Imagens RGB (PNG/JPG/TIF)
    │   ├── img_001.png
    │   ├── img_002.png
    │   └── ...
    └── masks/          # Mascaras binarias (0=fundo, 255=lago)
        ├── img_001.png
        ├── img_002.png
        └── ...

    Nomes dos arquivos em tiles/ e masks/ devem corresponder.
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

# Import condicional SAM / SAM-HQ
if Config.USE_SAM_HQ:
    try:
        from segment_anything_hq import sam_model_registry
        SAM_BACKEND = "segment_anything_hq"
    except ImportError:
        raise ImportError(
            "Config.USE_SAM_HQ=True, mas 'segment_anything_hq' nao encontrado.\n"
            "Instale: pip install segment-anything-hq\n"
            "Ou defina USE_SAM_HQ=False em config.py."
        )
else:
    from segment_anything import sam_model_registry
    SAM_BACKEND = "segment_anything"


# ============================================================================
# Dataset para dados de satelite
# ============================================================================

def normalize_image(image_tensor):
    """Normaliza com stats SAM (ImageNet)."""
    mean = torch.tensor(Config.PIXEL_MEAN, device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(Config.PIXEL_STD, device=image_tensor.device).view(1, 3, 1, 1)
    return (image_tensor - mean) / std


class SatelliteLakeDataset(Dataset):
    """Dataset generico para tiles + mascaras de satelite.

    Suporta tiles de qualquer tamanho (resize para 1024x1024).
    Mascaras sao binarizadas (> 127 = positivo).
    Inclui augmentacao basica (flips + rotacoes).
    """

    def __init__(self, tile_paths, mask_paths, augment=True):
        assert len(tile_paths) == len(mask_paths)
        self.tile_paths = tile_paths
        self.mask_paths = mask_paths
        self.augment = augment

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        # Carregar imagem
        image = cv2.imread(str(self.tile_paths[idx]))
        if image is None:
            # Fallback: imagem preta + mascara vazia
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            mask = np.zeros((512, 512), dtype=np.float32)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Carregar mascara
            mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
            else:
                mask = (mask > 127).astype(np.float32)

        # Augmentacao basica
        if self.augment:
            # Random flip horizontal
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            # Random flip vertical
            if random.random() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()
            # Random rot90
            k = random.randint(0, 3)
            if k > 0:
                image = np.rot90(image, k).copy()
                mask = np.rot90(mask, k).copy()
            # Random brightness/contrast
            if random.random() > 0.5:
                alpha = random.uniform(0.8, 1.2)  # contraste
                beta = random.uniform(-20, 20)      # brilho
                image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # Resize
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Tensores
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = normalize_image(image_tensor.unsqueeze(0)).squeeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor


# ============================================================================
# Loss (reutilizada do 03_finetune_sam.py)
# ============================================================================

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)
        target = target.view(-1).float()
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth + self.eps)
        tversky = torch.clamp(tversky, self.eps, 1.0)
        return torch.clamp(1.0 - tversky, min=self.eps) ** self.gamma


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        pt = torch.clamp(pt, self.eps, 1.0 - self.eps)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)
        return loss.mean()


class PretrainLoss(nn.Module):
    """Loss para pre-treino: Focal Tversky + Focal (foco em lakes)."""
    def __init__(self):
        super().__init__()
        self.tversky = FocalTverskyLoss(alpha=0.4, beta=0.6, gamma=0.75)
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)

    def forward(self, pred, target):
        return 0.5 * self.tversky(pred, target) + 0.5 * self.focal(pred, target)


# ============================================================================
# Coleta de dados
# ============================================================================

def collect_satellite_data(data_dir):
    """Coleta pares tile-mascara de um diretorio de dados de satelite.

    Args:
        data_dir: Path para o diretorio raiz com tiles/ e masks/.

    Returns:
        tile_paths, mask_paths: Listas de caminhos correspondentes.
    """
    data_dir = Path(data_dir)
    tiles_dir = data_dir / "tiles"
    masks_dir = data_dir / "masks"

    if not tiles_dir.exists():
        raise FileNotFoundError(f"Diretorio de tiles nao encontrado: {tiles_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Diretorio de mascaras nao encontrado: {masks_dir}")

    tile_paths = []
    mask_paths = []

    # Extensoes suportadas
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

    for tile_file in sorted(tiles_dir.iterdir()):
        if tile_file.suffix.lower() not in extensions:
            continue

        # Procurar mascara correspondente (mesmo nome, qualquer extensao)
        mask_found = None
        for ext in extensions:
            candidate = masks_dir / f"{tile_file.stem}{ext}"
            if candidate.exists():
                mask_found = candidate
                break

        if mask_found is not None:
            tile_paths.append(tile_file)
            mask_paths.append(mask_found)

    return tile_paths, mask_paths


def generate_prompts_pretrain(mask_tensor, device):
    """Gera prompts para pre-treino (centro da mascara ou ponto aleatorio).

    Simplificado em relacao ao 03_finetune_sam.py: usa apenas pontos.
    """
    batch_size = mask_tensor.shape[0]
    scale = 1024.0 / 256.0
    all_points = []
    all_labels = []

    for i in range(batch_size):
        mask_np = mask_tensor[i, 0].cpu().numpy()
        ys, xs = np.where(mask_np > 0.5)

        if len(xs) == 0:
            # Negativo: ponto aleatorio
            px = random.uniform(64, 960)
            py = random.uniform(64, 960)
            all_points.append([[px, py]])
            all_labels.append([1])
        else:
            # Positivo: ponto aleatorio dentro da mascara + negativo fora
            pts, lbls = [], []
            idx = random.randint(0, len(xs) - 1)
            pts.append([float(xs[idx]) * scale, float(ys[idx]) * scale])
            lbls.append(1)

            bg_ys, bg_xs = np.where(mask_np <= 0.5)
            if len(bg_xs) > 0:
                neg_idx = random.randint(0, len(bg_xs) - 1)
                pts.append([float(bg_xs[neg_idx]) * scale, float(bg_ys[neg_idx]) * scale])
                lbls.append(0)

            all_points.append(pts)
            all_labels.append(lbls)

    # Pad
    max_pts = max(len(p) for p in all_points)
    for i in range(batch_size):
        while len(all_points[i]) < max_pts:
            all_points[i].append([0.0, 0.0])
            all_labels[i].append(-1)

    points_tensor = torch.tensor(all_points, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(all_labels, dtype=torch.int, device=device)
    return points_tensor, labels_tensor


# ============================================================================
# Treinamento
# ============================================================================

def pretrain_satellite(data_dir, epochs=20, lr=5e-5, val_split=0.2, max_samples=None):
    """Pre-treina o decoder SAM com dados de satelite de lagos supraglaciais.

    Estrategia: encoding on-the-fly (encoder float16, batch_size=1).
    O decoder aprende a segmentar lagos em imagens de satelite, servindo
    como warm-start para o fine-tuning com dados de drone.

    Args:
        data_dir: Diretorio com tiles/ e masks/.
        epochs: Numero de epocas.
        lr: Learning rate.
        val_split: Fracao para validacao.
        max_samples: Limitar numero de amostras (None = usar todas).
    """
    print(f"\n{'='*60}")
    print("PRE-TREINO COM DADOS DE SATELITE")
    print(f"{'='*60}")

    # Coletar dados
    tile_paths, mask_paths = collect_satellite_data(data_dir)
    print(f"  Pares tile-mascara encontrados: {len(tile_paths)}")

    if len(tile_paths) == 0:
        print("[ERRO] Nenhum par tile-mascara encontrado!")
        print(f"  Esperado: {data_dir}/tiles/*.png e {data_dir}/masks/*.png")
        return

    # Limitar amostras se solicitado
    if max_samples and len(tile_paths) > max_samples:
        random.seed(42)
        indices = random.sample(range(len(tile_paths)), max_samples)
        tile_paths = [tile_paths[i] for i in indices]
        mask_paths = [mask_paths[i] for i in indices]
        print(f"  Limitado a {max_samples} amostras")

    # Filtrar amostras positivas (com pelo menos algum lago)
    positive_tiles, positive_masks = [], []
    negative_tiles, negative_masks = [], []

    print("  Analisando mascaras...")
    for tp, mp in tqdm(zip(tile_paths, mask_paths), total=len(tile_paths), desc="  Filtrando"):
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.max() > 127:
            positive_tiles.append(tp)
            positive_masks.append(mp)
        else:
            negative_tiles.append(tp)
            negative_masks.append(mp)

    print(f"  Positivos (com lago): {len(positive_tiles)}")
    print(f"  Negativos (sem lago): {len(negative_tiles)}")

    # Balancear: mesma qtd de positivos e negativos
    n_neg = min(len(negative_tiles), len(positive_tiles))
    if n_neg > 0:
        random.seed(42)
        neg_idx = random.sample(range(len(negative_tiles)), n_neg)
        negative_tiles = [negative_tiles[i] for i in neg_idx]
        negative_masks = [negative_masks[i] for i in neg_idx]

    all_tiles = positive_tiles + negative_tiles
    all_masks = positive_masks + negative_masks
    print(f"  Total balanceado: {len(all_tiles)} ({len(positive_tiles)} pos + {n_neg} neg)")

    # Split treino/validacao
    indices = list(range(len(all_tiles)))
    random.seed(42)
    random.shuffle(indices)
    split_idx = int(len(indices) * (1 - val_split))

    train_tiles = [all_tiles[i] for i in indices[:split_idx]]
    train_masks = [all_masks[i] for i in indices[:split_idx]]
    val_tiles = [all_tiles[i] for i in indices[split_idx:]]
    val_masks = [all_masks[i] for i in indices[split_idx:]]
    print(f"  Treino: {len(train_tiles)} | Validacao: {len(val_tiles)}")

    # Datasets
    train_dataset = SatelliteLakeDataset(train_tiles, train_masks, augment=True)
    val_dataset = SatelliteLakeDataset(val_tiles, val_masks, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Carregar SAM completo
    sam_variant = "SAM-HQ" if Config.USE_SAM_HQ else "SAM"
    print(f"\n  Carregando {sam_variant} ({Config.MODEL_TYPE})...")
    sam = sam_model_registry[Config.MODEL_TYPE](checkpoint=str(Config.SAM_CHECKPOINT))
    sam.to(Config.DEVICE)

    # Encoder congelado em float16
    sam.image_encoder.half()
    sam.image_encoder.eval()
    for param in sam.image_encoder.parameters():
        param.requires_grad = False

    # Decoder e prompt encoder treinaveis
    sam.mask_decoder.train()
    sam.prompt_encoder.train()
    trainable_params = list(sam.mask_decoder.parameters()) + list(sam.prompt_encoder.parameters())
    for param in trainable_params:
        param.requires_grad = True

    total_params = sum(p.numel() for p in trainable_params)
    print(f"  Params treinaveis: {total_params:,} ({total_params/1e6:.1f}M)")

    # Optimizer e scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    warmup_epochs = min(3, epochs // 3)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = PretrainLoss()
    image_pe = sam.prompt_encoder.get_dense_pe()

    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_dice = 0.0
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    print(f"\n  Iniciando pre-treino ({epochs} epocas, LR={lr})...\n", flush=True)

    for epoch in range(1, epochs + 1):
        # --- TREINO ---
        sam.mask_decoder.train()
        sam.prompt_encoder.train()
        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"  Epoca {epoch:02d}/{epochs} [treino]")
        for images, masks_gt in pbar:
            images = images.to(Config.DEVICE)
            masks_gt = masks_gt.to(Config.DEVICE)

            optimizer.zero_grad()

            # Encoder (float16, sem grad)
            with torch.no_grad():
                img_fp16 = images.half()
                if Config.USE_SAM_HQ:
                    emb, interm = sam.image_encoder(img_fp16)
                    emb = emb.float()
                    interm = [e.float() for e in interm]
                else:
                    emb = sam.image_encoder(img_fp16).float()
                    interm = None
                del img_fp16

            # Prompts
            pts, lbls = generate_prompts_pretrain(masks_gt, Config.DEVICE)
            sparse_emb, dense_emb = sam.prompt_encoder(
                points=(pts, lbls), boxes=None, masks=None,
            )

            # Decoder
            decoder_kwargs = dict(
                image_embeddings=emb,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            if Config.USE_SAM_HQ and interm is not None:
                decoder_kwargs["hq_token_only"] = True
                decoder_kwargs["interm_embeddings"] = interm

            low_res_mask, _ = sam.mask_decoder(**decoder_kwargs)

            loss = criterion(low_res_mask, masks_gt)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                train_loss_sum += loss.item()
                train_count += 1

            del emb, low_res_mask
            if interm is not None:
                del interm
            torch.cuda.empty_cache()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss_avg = train_loss_sum / max(train_count, 1)

        # --- VALIDACAO ---
        sam.mask_decoder.eval()
        sam.prompt_encoder.eval()
        val_loss_sum, val_dice_sum, val_count = 0.0, 0.0, 0

        with torch.no_grad():
            for images, masks_gt in val_loader:
                images = images.to(Config.DEVICE)
                masks_gt = masks_gt.to(Config.DEVICE)

                img_fp16 = images.half()
                if Config.USE_SAM_HQ:
                    emb, interm = sam.image_encoder(img_fp16)
                    emb = emb.float()
                    interm = [e.float() for e in interm]
                else:
                    emb = sam.image_encoder(img_fp16).float()
                    interm = None
                del img_fp16

                pts, lbls = generate_prompts_pretrain(masks_gt, Config.DEVICE)
                sparse_emb, dense_emb = sam.prompt_encoder(
                    points=(pts, lbls), boxes=None, masks=None,
                )

                decoder_kwargs = dict(
                    image_embeddings=emb,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                if Config.USE_SAM_HQ and interm is not None:
                    decoder_kwargs["hq_token_only"] = True
                    decoder_kwargs["interm_embeddings"] = interm

                low_res_mask, _ = sam.mask_decoder(**decoder_kwargs)

                loss = criterion(low_res_mask, masks_gt)
                if torch.isfinite(loss):
                    val_loss_sum += loss.item()

                pred_binary = (torch.sigmoid(low_res_mask) > 0.5).float()
                intersection = (pred_binary * masks_gt).sum()
                union = pred_binary.sum() + masks_gt.sum()
                dice = (2.0 * intersection + 1.0) / (union + 1.0)
                val_dice_sum += dice.item()
                val_count += 1

                del emb, low_res_mask
                if interm is not None:
                    del interm
                torch.cuda.empty_cache()

        val_loss_avg = val_loss_sum / max(val_count, 1)
        val_dice_avg = val_dice_sum / max(val_count, 1)
        scheduler.step()

        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss_avg)
        history["val_dice"].append(val_dice_avg)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoca {epoch:02d}/{epochs} | "
              f"Train Loss: {train_loss_avg:.4f} | "
              f"Val Loss: {val_loss_avg:.4f} | "
              f"Val Dice: {val_dice_avg:.4f} | "
              f"LR: {current_lr:.2e}", flush=True)

        # Salvar historico parcial a cada epoca (permite monitorar progresso)
        history_path = Config.MODELS_DIR / "pretrain_satellite_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Salvar melhor modelo
        if val_dice_avg > best_val_dice:
            best_val_dice = val_dice_avg
            save_path = Config.MODELS_DIR / "pretrained_satellite_lakes.pth"
            torch.save({
                "source": "satellite_pretrain",
                "data_dir": str(data_dir),
                "n_train": len(train_tiles),
                "n_val": len(val_tiles),
                "epoch": epoch,
                "val_dice": val_dice_avg,
                "val_loss": val_loss_avg,
                "sam_variant": "hq" if Config.USE_SAM_HQ else "standard",
                "mask_decoder_state_dict": sam.mask_decoder.state_dict(),
                "prompt_encoder_state_dict": sam.prompt_encoder.state_dict(),
                "history": history,
            }, save_path)
            print(f"    -> Melhor modelo salvo: {save_path.name} (Dice: {val_dice_avg:.4f})", flush=True)

    # Liberar GPU
    del sam
    torch.cuda.empty_cache()

    # Salvar historico
    history_path = Config.MODELS_DIR / "pretrain_satellite_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Pre-treino concluido!")
    print(f"  Melhor Val Dice: {best_val_dice:.4f}")
    print(f"  Checkpoint salvo em: {Config.MODELS_DIR / 'pretrained_satellite_lakes.pth'}")
    print(f"\n  Proximo passo: fine-tunar com dados de drone:")
    print(f"    python 03_finetune_sam.py --feature lakes --pretrained models/pretrained_satellite_lakes.pth")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pre-treino do decoder SAM com dados de satelite"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Diretorio com tiles/ e masks/ do dataset de satelite"
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Numero de epocas (default: 20)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Fracao para validacao (default: 0.2)"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Limitar numero de amostras (default: usar todas)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("PRE-TREINO SAM COM DADOS DE SATELITE")
    print("=" * 60)
    print(f"Dataset: {args.data_dir}")
    print(f"Device: {Config.DEVICE}")
    print(f"Backend: {SAM_BACKEND}")
    print(f"Epocas: {args.epochs}")
    print(f"LR: {args.lr}", flush=True)

    start = time.time()
    pretrain_satellite(
        args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        max_samples=args.max_samples,
    )
    elapsed = time.time() - start
    print(f"\nTempo total: {elapsed/60:.1f} minutos")


if __name__ == "__main__":
    main()
