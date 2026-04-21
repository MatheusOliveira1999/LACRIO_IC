"""
03_finetune_sam.py - Fine-tuning do SAM para feições supraglaciais

Projeto: LACRIO IC - Extração de Feições Supraglaciais
Fase 3: Fine-tuning do decoder do SAM com anotações manuais

Estratégia de memória (GPU <= 4GB):
  Modo embeddings (padrão):
    1. Pré-computa embeddings do encoder em float16, 1 imagem por vez
    2. Salva embeddings em disco (~150 arquivos de ~1MB cada)
    3. Treina APENAS o decoder usando embeddings em cache (encoder fora da GPU)
    Consumo estimado: ~1-2 GB VRAM durante o treino

  Modo on-the-fly (--augment e/ou --lora):
    1. Encoder roda durante o treino em float16 (1 imagem por vez)
    2. Augmentacoes aplicadas na imagem antes do encoder
    3. LoRA injeta params treinaveis nas projecoes QKV do encoder
    Consumo estimado: ~3-4 GB VRAM (batch_size=1 forcado)

Uso:
    python 03_finetune_sam.py                          # Treina todas as feições
    python 03_finetune_sam.py --feature lakes           # Treina apenas lagos
    python 03_finetune_sam.py --feature crevasses       # Treina apenas crevasses
    python 03_finetune_sam.py --epochs 30               # Customizar épocas
    python 03_finetune_sam.py --augment                 # Com data augmentation
    python 03_finetune_sam.py --augment --lora           # Augmentation + LoRA
    python 03_finetune_sam.py --lora                    # LoRA (implica --augment)
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
    except ModuleNotFoundError as exc:
        if exc.name and exc.name != "segment_anything_hq":
            raise ImportError(
                f"Config.USE_SAM_HQ=True, mas a dependencia '{exc.name}' esta faltando.\n"
                f"Instale no ambiente atual: pip install {exc.name}\n"
                "Ou defina USE_SAM_HQ=False em config.py."
            ) from exc
        raise ImportError(
            "Config.USE_SAM_HQ=True, mas o modulo 'segment_anything_hq' nao foi encontrado.\n"
            "Instale no ambiente atual: pip install segment-anything-hq\n"
            "Ou defina USE_SAM_HQ=False em config.py."
        ) from exc
    except ImportError as exc:
        raise ImportError(
            "Falha ao importar 'segment_anything_hq'. Verifique dependencias do pacote.\n"
            "Ou defina USE_SAM_HQ=False em config.py."
        ) from exc
else:
    from segment_anything import sam_model_registry
    from segment_anything.modeling.mask_decoder import MaskDecoder
    from segment_anything.modeling.prompt_encoder import PromptEncoder
    from segment_anything.modeling.transformer import TwoWayTransformer
    SAM_BACKEND = "segment_anything"


# ============================================================================
# Diretório de cache para embeddings
# ============================================================================
EMBEDDINGS_CACHE_DIR = Config.PROJECT_DIR / "embeddings_cache"


def build_decoder_and_prompt_encoder(checkpoint_path, device):
    """Constrói decoder e prompt encoder SEM carregar o image encoder.

    Para SAM padrão: constrói módulos diretamente (mais eficiente em memória).
    Para SAM-HQ: carrega modelo completo e extrai decoder/prompt_encoder
    (necessário porque MaskDecoderHQ tem arquitetura diferente com HQ tokens).

    Args:
        checkpoint_path: Caminho para o checkpoint SAM completo.
        device: Dispositivo (cuda/cpu).

    Returns:
        mask_decoder, prompt_encoder: Módulos prontos com pesos carregados.
    """
    if Config.USE_SAM_HQ:
        # SAM-HQ: carregar modelo completo e extrair componentes
        # MaskDecoderHQ tem componentes extras (hf_token, hf_mlp, compress_vit_feat, etc.)
        print("  Carregando SAM-HQ para extrair decoder + prompt encoder...")
        sam = sam_model_registry[Config.MODEL_TYPE](
            checkpoint=str(checkpoint_path)
        )
        mask_decoder = sam.mask_decoder
        prompt_encoder = sam.prompt_encoder

        # Liberar image encoder (não necessário durante treino com embeddings em cache)
        del sam.image_encoder
        del sam
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        mask_decoder.to(device)
        prompt_encoder.to(device)

        return mask_decoder, prompt_encoder
    else:
        # SAM padrão: construir módulos diretamente (sem image encoder)
        mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(64, 64),
            input_image_size=(1024, 1024),
            mask_in_chans=16,
        )

        # Carregar apenas os pesos relevantes do checkpoint
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        decoder_state = {k.replace("mask_decoder.", ""): v
                         for k, v in state_dict.items() if k.startswith("mask_decoder.")}
        prompt_state = {k.replace("prompt_encoder.", ""): v
                        for k, v in state_dict.items() if k.startswith("prompt_encoder.")}

        mask_decoder.load_state_dict(decoder_state)
        prompt_encoder.load_state_dict(prompt_state)

        mask_decoder.to(device)
        prompt_encoder.to(device)

        return mask_decoder, prompt_encoder


# ============================================================================
# Fase 1: Pré-computar embeddings do encoder
# ============================================================================

def normalize_image(image_tensor):
    """Aplica normalização SAM (ImageNet stats) a um tensor de imagem.

    Args:
        image_tensor: Tensor (B, 3, H, W) float com pixels [0, 255].

    Returns:
        Tensor normalizado (B, 3, H, W).
    """
    mean = torch.tensor(Config.PIXEL_MEAN, device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(Config.PIXEL_STD, device=image_tensor.device).view(1, 3, 1, 1)
    return (image_tensor - mean) / std


def precompute_embeddings(tile_paths, force=False):
    """Pré-computa image embeddings usando o encoder em float16.

    Roda o encoder pesado (91M params) 1 imagem por vez em float16,
    salva os embeddings em disco, e libera o encoder da GPU.

    Args:
        tile_paths: Lista de caminhos para tiles PNG.
        force: Se True, recomputa mesmo se já existem em cache.

    Returns:
        embedding_paths: Lista de caminhos para embeddings salvos (.pt).
    """
    EMBEDDINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Verificar quais já existem no cache
    to_compute = []
    embedding_paths = []
    for tile_path in tile_paths:
        cache_key = f"{tile_path.parent.name}_{tile_path.stem}"
        emb_path = EMBEDDINGS_CACHE_DIR / f"{cache_key}.pt"
        embedding_paths.append(emb_path)
        if force or not emb_path.exists():
            to_compute.append((tile_path, emb_path))

    if not to_compute:
        print(f"  Embeddings em cache: {len(tile_paths)}/{len(tile_paths)} (reutilizando)")
        # Verificar compatibilidade do cache com modo SAM atual
        sample = torch.load(embedding_paths[0], map_location="cpu", weights_only=True)
        is_hq_cache = isinstance(sample, dict) and "interm" in sample
        if Config.USE_SAM_HQ and not is_hq_cache:
            print("  [AVISO] Cache de SAM padrao mas USE_SAM_HQ=True! "
                  "Recompute com force=True ou delete embeddings_cache/")
        elif not Config.USE_SAM_HQ and is_hq_cache:
            print("  [AVISO] Cache de SAM-HQ mas USE_SAM_HQ=False! "
                  "Recompute com force=True ou delete embeddings_cache/")
        del sample
        return embedding_paths

    print(f"  Embeddings a computar: {len(to_compute)} (cache: {len(tile_paths) - len(to_compute)})")

    # Carregar encoder em float16
    sam_variant = "SAM-HQ" if Config.USE_SAM_HQ else "SAM"
    print(f"  Carregando {sam_variant} encoder ({Config.MODEL_TYPE}) em float16...")
    sam = sam_model_registry[Config.MODEL_TYPE](checkpoint=str(Config.SAM_CHECKPOINT))
    sam.image_encoder.to(Config.DEVICE).half()
    sam.image_encoder.eval()

    # Processar 1 imagem por vez
    with torch.no_grad():
        for tile_path, emb_path in tqdm(to_compute, desc="  Pré-computando embeddings"):
            image = cv2.imread(str(tile_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)

            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
            image_tensor = image_tensor.unsqueeze(0)

            # Normalizar com stats SAM/ImageNet ANTES de converter para float16
            image_tensor = normalize_image(image_tensor)
            image_tensor = image_tensor.to(Config.DEVICE).half()

            if Config.USE_SAM_HQ:
                # SAM-HQ retorna tupla: (embedding, interm_embeddings)
                embedding, interm_embeddings = sam.image_encoder(image_tensor)
                # Salvar embedding + features intermediarias para o decoder HQ
                torch.save({
                    "embedding": embedding.float().cpu(),
                    "interm": interm_embeddings[0].float().cpu(),
                }, emb_path)
                del image_tensor, embedding, interm_embeddings
            else:
                embedding = sam.image_encoder(image_tensor)
                # Salvar em float32 para treino estável do decoder
                torch.save(embedding.float().cpu(), emb_path)
                del image_tensor, embedding

            # Limpar VRAM a cada iteração
            torch.cuda.empty_cache()

    # Liberar encoder da GPU completamente
    del sam
    torch.cuda.empty_cache()
    print(f"  Encoder liberado da GPU. Embeddings salvos em: {EMBEDDINGS_CACHE_DIR}")

    return embedding_paths


# ============================================================================
# Dataset (usa embeddings pré-computados)
# ============================================================================

class GlacierEmbeddingDataset(Dataset):
    """Dataset que carrega embeddings pré-computados + máscaras.

    Suporta amostras negativas (mask_path=None → máscara toda zeros).
    Suporta SAM padrão (tensor) e SAM-HQ (dict com embedding + interm).
    """

    def __init__(self, embedding_paths, mask_paths):
        assert len(embedding_paths) == len(mask_paths)
        self.embedding_paths = embedding_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.embedding_paths)

    def __getitem__(self, idx):
        # Carregar embedding pré-computado (já em float32)
        raw = torch.load(self.embedding_paths[idx], weights_only=True)

        if isinstance(raw, dict):
            # SAM-HQ: dict com embedding + features intermediarias
            embedding = raw["embedding"].squeeze(0)  # (256, 64, 64)
            interm = raw["interm"].squeeze(0)          # (64, 64, 768)
        else:
            # SAM padrão: tensor direto
            embedding = raw.squeeze(0)                 # (256, 64, 64)
            interm = torch.empty(0)                    # placeholder

        # Carregar máscara binária (ou zeros para amostras negativas)
        if self.mask_paths[idx] is not None:
            mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
            mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        else:
            mask = np.zeros((256, 256), dtype=np.float32)

        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, 256, 256)

        return embedding, mask_tensor, interm


# ============================================================================
# Dataset on-the-fly (Etapa 3: augmentacao + Etapa 4: LoRA)
# ============================================================================

def get_train_transform():
    """Retorna pipeline de augmentacao para treino.

    Requer albumentations instalado (pip install albumentations).
    Transformacoes geometricas aplicam na imagem + mascara.
    Transformacoes espectrais aplicam apenas na imagem.

    Returns:
        Compose de augmentacoes Albumentations.
    """
    import albumentations as A

    return A.Compose([
        # Geometricas (imagem + mascara)
        A.RandomRotate90(p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101, p=0.5
        ),
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),

        # Espectrais (apenas imagem)
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        A.RandomGamma(gamma_limit=(70, 150), p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
    ])


class GlacierOnTheFlyDataset(Dataset):
    """Dataset que carrega imagens + mascaras com augmentacao on-the-fly.

    Diferente de GlacierEmbeddingDataset, retorna imagem RGB (nao embedding).
    O encoder roda durante o treino para permitir augmentacao e LoRA.
    Suporta amostras negativas (mask_path=None -> mascara toda zeros).
    Suporta Copy-Paste augmentation: cola feicoes de outros tiles positivos.

    Args:
        tile_paths: Lista de caminhos para tiles PNG.
        mask_paths: Lista de caminhos para mascaras (ou None para negativos).
        transform: Pipeline Albumentations (None para validacao).
        use_copy_paste: Se True, aplica copy-paste augmentation.
        copy_paste_prob: Probabilidade de aplicar copy-paste por amostra.
    """

    def __init__(self, tile_paths, mask_paths, transform=None,
                 use_copy_paste=False, copy_paste_prob=0.5):
        assert len(tile_paths) == len(mask_paths)
        self.tile_paths = tile_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.use_copy_paste = use_copy_paste
        self.copy_paste_prob = copy_paste_prob

        # Indices dos tiles positivos (para copy-paste source)
        self.positive_indices = [
            i for i, m in enumerate(mask_paths) if m is not None
        ]

    def __len__(self):
        return len(self.tile_paths)

    def _apply_copy_paste(self, image, mask):
        """Cola uma feicao de outro tile positivo nesta imagem.

        Seleciona aleatoriamente um tile positivo, extrai a regiao da feicao,
        e cola na imagem atual em posicao aleatoria. Atualiza a mascara.
        Ref: Ghiasi et al. (2021) "Simple Copy-Paste is a Strong Data Augmentation"
        """
        if not self.positive_indices or random.random() > self.copy_paste_prob:
            return image, mask

        # Selecionar tile fonte aleatorio
        src_idx = random.choice(self.positive_indices)
        src_img = cv2.imread(str(self.tile_paths[src_idx]))
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        src_mask = cv2.imread(str(self.mask_paths[src_idx]), cv2.IMREAD_GRAYSCALE)
        src_mask = (src_mask > 127).astype(np.uint8)

        # Encontrar bounding box da feicao no tile fonte
        ys, xs = np.where(src_mask > 0)
        if len(xs) == 0:
            return image, mask

        x1, x2 = xs.min(), xs.max() + 1
        y1, y2 = ys.min(), ys.max() + 1

        # Recortar feicao + mascara
        crop_img = src_img[y1:y2, x1:x2].copy()
        crop_mask = src_mask[y1:y2, x1:x2].copy()

        ch, cw = crop_img.shape[:2]
        h, w = image.shape[:2]

        if ch >= h or cw >= w or ch == 0 or cw == 0:
            return image, mask

        # Posicao aleatoria no tile destino
        paste_y = random.randint(0, h - ch)
        paste_x = random.randint(0, w - cw)

        # Colar usando mascara como alpha (apenas pixels da feicao)
        alpha = crop_mask.astype(np.float32)
        for c_ch in range(3):
            image[paste_y:paste_y+ch, paste_x:paste_x+cw, c_ch] = (
                image[paste_y:paste_y+ch, paste_x:paste_x+cw, c_ch] * (1 - alpha) +
                crop_img[:, :, c_ch] * alpha
            ).astype(np.uint8)

        # Atualizar mascara
        mask[paste_y:paste_y+ch, paste_x:paste_x+cw] = np.maximum(
            mask[paste_y:paste_y+ch, paste_x:paste_x+cw],
            crop_mask.astype(np.float32)
        )

        return image, mask

    def __getitem__(self, idx):
        # Carregar imagem RGB
        image = cv2.imread(str(self.tile_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Carregar mascara binaria (ou zeros para amostras negativas)
        if self.mask_paths[idx] is not None:
            mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Copy-Paste augmentation (antes das outras augmentacoes)
        if self.use_copy_paste:
            image, mask = self._apply_copy_paste(image, mask)

        # Augmentacao (aplicar ANTES do resize para manter consistencia)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Resize: imagem para 1024x1024 (entrada SAM), mascara para 256x256 (saida decoder)
        image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        # Converter para tensores
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()  # (3, 1024, 1024)
        image_tensor = normalize_image(image_tensor.unsqueeze(0)).squeeze(0)  # normalizar
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # (1, 256, 256)

        return image_tensor, mask_tensor


# ============================================================================
# Loss Functions
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss para segmentação binária."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Dice Loss + Binary Cross-Entropy (como descrito no projeto)."""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.dice(pred, target) + self.bce(pred, target)


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss com controle independente de FP/FN.

    Permite ajustar o balanco entre falsos positivos (alpha) e
    falsos negativos (beta) por feicao. O fator gamma focaliza
    em exemplos dificeis (hard examples).

    Ref: Abraham & Khan (2019) - A Novel Focal Tversky Loss
    """

    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha   # peso para falsos positivos
        self.beta = beta     # peso para falsos negativos
        self.gamma = gamma   # fator de focalizacao (< 1 = foco em hard examples)
        self.smooth = smooth
        self.eps = eps

    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        pred = torch.clamp(pred, self.eps, 1.0 - self.eps)
        target = target.view(-1).float()

        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth + self.eps
        )
        tversky = torch.clamp(tversky, self.eps, 1.0)

        # Evita base negativa com gamma fracionario (causa NaN)
        focal_term = torch.clamp(1.0 - tversky, min=self.eps)
        return focal_term ** self.gamma


class FocalLoss(nn.Module):
    """Binary Focal Loss para lidar com desbalanco de classes.

    Reduz contribuicao de exemplos faceis (background = gelo) e
    foca em hard examples (bordas, features ambiguas).

    Ref: Lin et al. (2017) - Focal Loss for Dense Object Detection
    """

    def __init__(self, alpha=0.75, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)  # probabilidade do label correto
        pt = torch.clamp(pt, self.eps, 1.0 - self.eps)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)
        return loss.mean()


class ImprovedLoss(nn.Module):
    """Loss combinada Focal Tversky + Focal, com pesos por feicao.

    Lakes: penaliza mais FP (alpha=0.6) para reduzir deteccao de sombras.
    Crevasses/Channels: penaliza mais FN (beta=0.7) para melhorar recall.
    """

    def __init__(self, feature="lakes"):
        super().__init__()
        if feature == "lakes":
            self.tversky = FocalTverskyLoss(alpha=0.6, beta=0.4, gamma=0.75)
        else:  # crevasses, channels
            self.tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)

    def forward(self, pred, target):
        return 0.5 * self.tversky(pred, target) + 0.5 * self.focal(pred, target)


# ============================================================================
# LoRA - Low-Rank Adaptation (Etapa 4)
# ============================================================================

class LoRALinear(nn.Module):
    """Camada Linear com adaptacao LoRA (Low-Rank Adaptation).

    Substitui uma camada Linear existente adicionando matrizes de baixo rank
    A (down-projection) e B (up-projection). O peso original fica congelado.

    Saida: original(x) + B(A(dropout(x))) * scale

    Ref: Hu et al. (2022) "LoRA: Low-Rank Adaptation of Large Language Models"

    Args:
        original_layer: Camada nn.Linear original a ser adaptada.
        r: Rank das matrizes LoRA.
        alpha: Fator de escala (scale = alpha / r).
        dropout: Taxa de dropout antes da projecao LoRA.
    """

    def __init__(self, original_layer, r=4, alpha=16, dropout=0.1):
        super().__init__()
        self.original = original_layer
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        self.scale = alpha / r

        # Matrizes LoRA
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        # Inicializacao: A com Kaiming, B com zeros (LoRA comeca como identidade)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

        # Congelar peso original
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        original_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scale
        return original_out + lora_out


def inject_lora(image_encoder, r=4, alpha=16, dropout=0.1):
    """Injeta camadas LoRA nas projecoes QKV de atencao do ViT encoder.

    Percorre todos os blocos transformer e substitui a camada `attn.qkv`
    (projecao conjunta Query-Key-Value) por uma versao LoRA.

    Args:
        image_encoder: Encoder ViT do SAM (sam.image_encoder).
        r: Rank LoRA.
        alpha: Fator de escala.
        dropout: Taxa de dropout.

    Returns:
        n_injected: Numero de camadas LoRA injetadas.
    """
    n_injected = 0

    for block in image_encoder.blocks:
        if hasattr(block.attn, "qkv") and isinstance(block.attn.qkv, nn.Linear):
            original_qkv = block.attn.qkv
            block.attn.qkv = LoRALinear(original_qkv, r=r, alpha=alpha, dropout=dropout)
            n_injected += 1

    return n_injected


# ============================================================================
# Funções auxiliares
# ============================================================================

def collect_pairs(feature: str, years=None, neg_ratio=1.0, shadow_neg_ratio=0.5):
    """Coleta pares tile-máscara para uma feição, incluindo amostras negativas.

    Amostras negativas incluem:
    - Hard negatives de sombra: tiles com alta cobertura de sombra topográfica
      (ensina o modelo a NÃO segmentar sombras como lagos)
    - Negativos aleatórios: tiles sem anotação selecionados aleatoriamente

    Args:
        feature: Nome da feição (lakes, crevasses, channels).
        years: Lista de anos para usar. Se None, usa todos disponíveis.
        neg_ratio: Proporção de negativos em relação aos positivos (1.0 = igual).
        shadow_neg_ratio: Fração dos negativos que devem ser hard negatives de
                          sombra (0.5 = metade). Usado apenas para 'lakes'.

    Returns:
        tile_paths, mask_paths: Listas de caminhos (mask=None para negativos).
    """
    if years is None:
        years = Config.YEARS

    tile_paths = []
    mask_paths = []
    annotated_tile_ids = set()

    for year in years:
        annotations_dir = Config.MASKS_DIR / str(year) / "annotations" / feature
        if not annotations_dir.exists():
            continue

        for mask_file in sorted(annotations_dir.glob(f"tile_*_{feature}.png")):
            tile_id = mask_file.stem.replace(f"_{feature}", "")
            tile_file = Config.TILES_DIR / str(year) / f"{tile_id}.png"

            if tile_file.exists():
                tile_paths.append(tile_file)
                mask_paths.append(mask_file)
                annotated_tile_ids.add((year, tile_id))

    n_positives = len(tile_paths)
    if n_positives == 0:
        return tile_paths, mask_paths

    # Coletar amostras negativas (tiles sem anotação)
    n_negatives = int(n_positives * neg_ratio)

    # Hard negatives de sombra (apenas para lakes)
    shadow_negatives = []
    n_shadow_neg = 0
    if feature == "lakes" and shadow_neg_ratio > 0:
        n_shadow_neg = int(n_negatives * shadow_neg_ratio)
        shadow_negatives = _collect_shadow_negatives(
            years, n_shadow_neg, annotated_tile_ids
        )
        n_shadow_neg = len(shadow_negatives)  # Pode ser menor que solicitado

    # Negativos aleatórios para completar
    n_random_neg = n_negatives - n_shadow_neg
    all_negative_candidates = []

    shadow_paths_set = set(str(p) for p in shadow_negatives)

    for year in years:
        tiles_dir = Config.TILES_DIR / str(year)
        if not tiles_dir.exists():
            continue
        for tile_file in tiles_dir.glob("tile_*.png"):
            tile_id = tile_file.stem
            if ((year, tile_id) not in annotated_tile_ids
                    and str(tile_file) not in shadow_paths_set):
                all_negative_candidates.append(tile_file)

    random.seed(42)
    if len(all_negative_candidates) > n_random_neg:
        random_negatives = random.sample(all_negative_candidates, n_random_neg)
    else:
        random_negatives = all_negative_candidates

    # Adicionar hard negatives de sombra
    for tile_file in shadow_negatives:
        tile_paths.append(tile_file)
        mask_paths.append(None)

    # Adicionar negativos aleatórios
    for tile_file in random_negatives:
        tile_paths.append(tile_file)
        mask_paths.append(None)

    print(f"  Amostras positivas: {n_positives} | "
          f"Negativas sombra: {n_shadow_neg} | "
          f"Negativas aleatórias: {len(random_negatives)}")

    return tile_paths, mask_paths


def _collect_shadow_negatives(years, n_samples, annotated_ids):
    """Seleciona tiles com maior cobertura de sombra como hard negatives.

    Usa a máscara de sombra DEM para identificar tiles que contêm sombras
    significativas. Estes tiles ensinam o modelo a não confundir sombra com lago.

    Args:
        years: Lista de anos para buscar.
        n_samples: Número de hard negatives desejados.
        annotated_ids: Set de (year, tile_id) já anotados (excluir).

    Returns:
        Lista de Paths para tiles com maior cobertura de sombra.
    """
    from shadow_utils import get_shadow_coverage_for_tiles

    shadow_tiles = []

    for year in years:
        coverage = get_shadow_coverage_for_tiles(year)
        for tile_id, cov in coverage.items():
            if (year, tile_id) in annotated_ids:
                continue
            if cov < 0.05:  # Mínimo 5% de sombra
                continue
            tile_file = Config.TILES_DIR / str(year) / f"{tile_id}.png"
            if tile_file.exists():
                shadow_tiles.append((tile_file, cov))

    # Ordenar por cobertura de sombra (maior primeiro)
    shadow_tiles.sort(key=lambda x: x[1], reverse=True)

    selected = [t[0] for t in shadow_tiles[:n_samples]]
    if selected:
        print(f"  Hard negatives de sombra: {len(selected)} tiles "
              f"(cobertura {shadow_tiles[0][1]:.0%} - "
              f"{shadow_tiles[min(len(selected)-1, len(shadow_tiles)-1)][1]:.0%})")

    return selected


def split_dataset(tile_paths, mask_paths, train_ratio=0.8, seed=42):
    """Divide dataset em treino e validação."""
    indices = list(range(len(tile_paths)))
    random.seed(seed)
    random.shuffle(indices)

    split_idx = int(len(indices) * train_ratio)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    train_tiles = [tile_paths[i] for i in train_idx]
    train_masks = [mask_paths[i] for i in train_idx]
    val_tiles = [tile_paths[i] for i in val_idx]
    val_masks = [mask_paths[i] for i in val_idx]

    return train_tiles, train_masks, val_tiles, val_masks


def generate_prompts(mask_tensor, device, strategy="random"):
    """Gera pontos de prompt variados para treino robusto.

    Estratégias:
    - "center": ponto no centroide (compatível com versão anterior)
    - "random": ponto aleatório dentro da máscara + ponto negativo fora
    - "bbox": bounding box da máscara (sem pontos)
    - "grid": grade 8x8 de pontos cegos (simula inferência real)

    Para amostras negativas (máscara vazia), gera ponto aleatório como
    foreground prompt. O target continua sendo zeros, ensinando o modelo
    a NÃO segmentar quando não há feição.

    Args:
        mask_tensor: Tensor (B, 1, 256, 256) com a máscara GT.
        device: Dispositivo (cuda/cpu).
        strategy: Tipo de prompt a gerar.

    Returns:
        Se strategy == "bbox":
            boxes: Tensor (B, 4), points=None, labels=None
        Se strategy == "grid":
            points: Tensor (B, 64, 2), labels: Tensor (B, 64)
        Senão:
            points: Tensor (B, N, 2), labels: Tensor (B, N)
    """
    batch_size = mask_tensor.shape[0]
    scale = 1024.0 / 256.0

    # Verificar quais amostras são negativas (máscara vazia)
    is_negative = [(mask_tensor[i, 0].sum().item() < 1.0) for i in range(batch_size)]

    if strategy == "grid":
        # Simula inferencia real: seleciona 1 ponto aleatorio da grade 8x8.
        # Se o ponto cai sobre a mascara GT → target = GT (segmentar).
        # Se o ponto cai fora → target = zeros (nao segmentar).
        # Isso ensina o modelo QUANDO segmentar, nao so COMO.
        stride = 128
        coords = np.arange(stride // 2, 1024, stride)
        xx, yy = np.meshgrid(coords, coords)
        grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)

        all_points = []
        all_labels = []
        for i in range(batch_size):
            # Escolher ponto aleatorio do grid
            idx = np.random.randint(len(grid_pts))
            pt = grid_pts[idx]
            all_points.append([[float(pt[0]), float(pt[1])]])
            all_labels.append([1])

            # Se o ponto cai fora da mascara, zerar o target para este sample.
            # Isso e feito pelo caller ao verificar se o ponto esta na mascara.
            # Aqui marcamos com label especial para o caller poder detectar.
            mask_np = mask_tensor[i, 0].cpu().numpy()
            pt_in_mask_space = (int(pt[0] / scale), int(pt[1] / scale))
            px = min(max(pt_in_mask_space[0], 0), 255)
            py = min(max(pt_in_mask_space[1], 0), 255)
            if mask_np[py, px] < 0.5:
                # Ponto fora da feicao: modelo deve retornar mascara vazia
                # Zeramos o target na mascara para ensinar "nada aqui"
                mask_tensor[i, 0] = 0.0

        points_tensor = torch.tensor(all_points, dtype=torch.float32, device=device)
        labels_tensor = torch.tensor(all_labels, dtype=torch.int, device=device)
        return points_tensor, labels_tensor, None

    if strategy == "bbox":
        boxes = []
        for i in range(batch_size):
            mask_np = mask_tensor[i, 0].cpu().numpy()
            ys, xs = np.where(mask_np > 0.5)
            if len(xs) > 0:
                x1, x2 = float(xs.min()) * scale, float(xs.max()) * scale
                y1, y2 = float(ys.min()) * scale, float(ys.max()) * scale
                noise = np.random.uniform(-10, 10, 4)
                x1 = max(0, x1 + noise[0])
                y1 = max(0, y1 + noise[1])
                x2 = min(1024, x2 + noise[2])
                y2 = min(1024, y2 + noise[3])
            else:
                # Negativo: bbox aleatória (modelo deve retornar vazio)
                cx = np.random.uniform(200, 800)
                cy = np.random.uniform(200, 800)
                hw = np.random.uniform(100, 300)
                x1, y1 = cx - hw, cy - hw
                x2, y2 = cx + hw, cy + hw
            boxes.append([x1, y1, x2, y2])
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=device)
        return None, None, boxes_tensor

    all_points = []
    all_labels = []

    for i in range(batch_size):
        mask_np = mask_tensor[i, 0].cpu().numpy()
        ys, xs = np.where(mask_np > 0.5)
        bg_ys, bg_xs = np.where(mask_np <= 0.5)

        if is_negative[i]:
            # Amostra negativa: ponto aleatório como foreground
            # O modelo deve aprender a retornar máscara vazia
            px = np.random.uniform(64, 960)
            py = np.random.uniform(64, 960)
            all_points.append([[px, py]])
            all_labels.append([1])
            continue

        if strategy == "center":
            cx = float(xs.mean()) * scale
            cy = float(ys.mean()) * scale
            all_points.append([[cx, cy]])
            all_labels.append([1])

        elif strategy == "random":
            pts = []
            lbls = []
            # 1-3 pontos positivos aleatórios dentro da máscara
            n_pos = random.randint(1, min(3, len(xs)))
            idxs = np.random.choice(len(xs), n_pos, replace=False)
            for idx in idxs:
                pts.append([float(xs[idx]) * scale, float(ys[idx]) * scale])
                lbls.append(1)
            # 1 ponto negativo fora da máscara
            if len(bg_xs) > 0:
                neg_idx = np.random.randint(len(bg_xs))
                pts.append([float(bg_xs[neg_idx]) * scale, float(bg_ys[neg_idx]) * scale])
                lbls.append(0)

            all_points.append(pts)
            all_labels.append(lbls)

    # Pad to same length within batch
    max_pts = max(len(p) for p in all_points)
    for i in range(batch_size):
        while len(all_points[i]) < max_pts:
            all_points[i].append([0.0, 0.0])
            all_labels[i].append(-1)  # -1 = padding (ignorado pelo SAM)

    points_tensor = torch.tensor(all_points, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(all_labels, dtype=torch.int, device=device)

    return points_tensor, labels_tensor, None


# ============================================================================
# Treinamento
# ============================================================================

def train_feature(feature: str, epochs: int, batch_size: int, lr: float,
                   pretrained_path: str = None):
    """Treina o SAM para uma feição específica.

    Estratégia em 2 etapas:
    1. Pré-computa embeddings com encoder em float16 (1 por vez)
    2. Treina decoder usando embeddings em cache (encoder fora da GPU)

    Args:
        feature: Nome da feição.
        epochs: Número de épocas.
        batch_size: Tamanho do batch.
        lr: Learning rate.
    """
    print(f"\n{'='*60}")
    print(f"FINE-TUNING SAM - Feição: {feature.upper()}")
    print(f"{'='*60}")

    # Coletar dados
    tile_paths, mask_paths = collect_pairs(feature)

    if len(tile_paths) == 0:
        print(f"[ERRO] Nenhuma anotação encontrada para '{feature}'.")
        for year in Config.YEARS:
            d = Config.MASKS_DIR / str(year) / "annotations" / feature
            print(f"  {d} (existe: {d.exists()})")
        return

    print(f"  Amostras encontradas: {len(tile_paths)}")

    # Split treino/validação
    train_tiles, train_masks, val_tiles, val_masks = split_dataset(
        tile_paths, mask_paths, Config.TRAIN_VAL_SPLIT
    )
    print(f"  Treino: {len(train_tiles)} | Validação: {len(val_tiles)}")

    # --- ETAPA 1: Pré-computar embeddings ---
    print(f"\n  [Etapa 1/2] Pré-computando embeddings do encoder (float16)...")
    all_tile_paths = train_tiles + val_tiles
    all_mask_paths = train_masks + val_masks
    all_emb_paths = precompute_embeddings(all_tile_paths)

    train_emb = all_emb_paths[:len(train_tiles)]
    val_emb = all_emb_paths[len(train_tiles):]

    # --- ETAPA 2: Treinar decoder ---
    print(f"\n  [Etapa 2/2] Treinando mask decoder + prompt encoder...")

    # Criar datasets com embeddings
    train_dataset = GlacierEmbeddingDataset(train_emb, train_masks)
    val_dataset = GlacierEmbeddingDataset(val_emb, val_masks)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Carregar APENAS decoder e prompt encoder (sem image encoder na memória!)
    print(f"  Carregando decoder + prompt encoder em {Config.DEVICE}...")
    print(f"  (image encoder NÃO carregado - usando embeddings em cache)")
    mask_decoder, prompt_encoder = build_decoder_and_prompt_encoder(
        Config.SAM_CHECKPOINT, Config.DEVICE
    )

    # Carregar pesos pre-treinados com satelite (warm-start)
    if pretrained_path:
        print(f"  Carregando pesos pre-treinados de: {pretrained_path}")
        pretrained_ckpt = torch.load(pretrained_path, map_location=Config.DEVICE)
        mask_decoder.load_state_dict(pretrained_ckpt["mask_decoder_state_dict"])
        prompt_encoder.load_state_dict(pretrained_ckpt["prompt_encoder_state_dict"])
        print(f"  ✓ Warm-start aplicado (pre-treinado em satelite)")

    # Treinar decoder + prompt encoder
    trainable_params = (list(mask_decoder.parameters()) +
                        list(prompt_encoder.parameters()))
    for param in trainable_params:
        param.requires_grad = True

    total_params = sum(p.numel() for p in trainable_params)
    print(f"  Parâmetros treináveis: {total_params:,} ({total_params/1e6:.1f}M)")

    # Optimizer e loss
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    criterion = ImprovedLoss(feature=feature)

    # Scheduler: warmup linear + cosine annealing
    warmup_epochs = Config.WARMUP_EPOCHS
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear warmup
        # Cosine annealing apos warmup
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_dice = 0.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    # Obter positional encoding (constante, computar uma vez)
    image_pe = prompt_encoder.get_dense_pe()

    for epoch in range(1, epochs + 1):
        # --- TREINO ---
        mask_decoder.train()
        prompt_encoder.train()

        train_loss_sum = 0.0
        train_count = 0

        # Alternar estratégias de prompt para robustez
        strategies = ["random", "center", "bbox", "grid"]

        pbar = tqdm(train_loader, desc=f"  Época {epoch:02d}/{epochs} [treino]")
        for batch_idx, (embeddings, masks_gt, interm_feats) in enumerate(pbar):
            embeddings = embeddings.to(Config.DEVICE)
            masks_gt = masks_gt.to(Config.DEVICE)
            if Config.USE_SAM_HQ:
                interm_feats = interm_feats.to(Config.DEVICE)

            strategy = strategies[batch_idx % len(strategies)]

            optimizer.zero_grad()
            batch_loss_sum = 0.0
            valid_samples = 0
            bs = embeddings.size(0)

            for i in range(bs):
                emb_i = embeddings[i:i+1]     # (1, 256, 64, 64)
                mask_i = masks_gt[i:i+1]       # (1, 1, 256, 256)
                pts, lbls, boxes = generate_prompts(mask_i, Config.DEVICE, strategy)

                if boxes is not None:
                    sparse_emb, dense_emb = prompt_encoder(
                        points=None, boxes=boxes, masks=None,
                    )
                else:
                    sparse_emb, dense_emb = prompt_encoder(
                        points=(pts, lbls), boxes=None, masks=None,
                    )

                decoder_kwargs = dict(
                    image_embeddings=emb_i,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                if Config.USE_SAM_HQ:
                    decoder_kwargs["hq_token_only"] = True
                    decoder_kwargs["interm_embeddings"] = [interm_feats[i:i+1]]

                low_res_mask, _ = mask_decoder(**decoder_kwargs)

                loss_i = criterion(low_res_mask, mask_i)
                if not torch.isfinite(loss_i):
                    continue

                (loss_i / bs).backward()
                batch_loss_sum += loss_i.item()
                valid_samples += 1

            if valid_samples > 0:
                # Gradient clipping para estabilidade
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            train_loss_sum += batch_loss_sum
            train_count += valid_samples
            mean_batch_loss = batch_loss_sum / max(valid_samples, 1)
            pbar.set_postfix(loss=f"{mean_batch_loss:.4f}", prompt=strategy)

        train_loss_avg = train_loss_sum / max(train_count, 1)

        # --- VALIDAÇÃO ---
        mask_decoder.eval()
        prompt_encoder.eval()
        val_loss_sum = 0.0
        val_loss_count = 0
        val_dice_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for embeddings, masks_gt, interm_feats in val_loader:
                embeddings = embeddings.to(Config.DEVICE)
                masks_gt = masks_gt.to(Config.DEVICE)
                if Config.USE_SAM_HQ:
                    interm_feats = interm_feats.to(Config.DEVICE)

                bs = embeddings.size(0)
                for i in range(bs):
                    emb_i = embeddings[i:i+1]
                    mask_i = masks_gt[i:i+1]
                    # Validação usa centro (determinístico)
                    pts, lbls, _ = generate_prompts(mask_i, Config.DEVICE, "center")

                    sparse_emb, dense_emb = prompt_encoder(
                        points=(pts, lbls), boxes=None, masks=None,
                    )

                    decoder_kwargs = dict(
                        image_embeddings=emb_i,
                        image_pe=image_pe,
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False,
                    )
                    if Config.USE_SAM_HQ:
                        decoder_kwargs["hq_token_only"] = True
                        decoder_kwargs["interm_embeddings"] = [interm_feats[i:i+1]]

                    low_res_mask, _ = mask_decoder(**decoder_kwargs)

                    loss = criterion(low_res_mask, mask_i)
                    if torch.isfinite(loss):
                        val_loss_sum += loss.item()
                        val_loss_count += 1

                    pred_binary = (torch.sigmoid(low_res_mask) > 0.5).float()
                    intersection = (pred_binary * mask_i).sum()
                    union_val = pred_binary.sum() + mask_i.sum()
                    dice = (2.0 * intersection + 1.0) / (union_val + 1.0)
                    val_dice_sum += dice.item()

                    val_count += 1

        val_loss_avg = val_loss_sum / max(val_loss_count, 1)
        val_dice_avg = val_dice_sum / max(val_count, 1)

        scheduler.step()

        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss_avg)
        history["val_dice"].append(val_dice_avg)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  Época {epoch:02d}/{epochs} | "
              f"Train Loss: {train_loss_avg:.4f} | "
              f"Val Loss: {val_loss_avg:.4f} | "
              f"Val Dice: {val_dice_avg:.4f} | "
              f"LR: {current_lr:.2e}")

        # Salvar melhor modelo (por val_dice, mais robusto que val_loss)
        if val_dice_avg > best_val_dice + Config.EARLY_STOPPING_MIN_DELTA:
            best_val_dice = val_dice_avg
            epochs_without_improvement = 0
            best_path = Config.MODELS_DIR / f"sam_finetuned_{feature}_best.pth"
            torch.save({
                "feature": feature,
                "epoch": epoch,
                "sam_variant": "hq" if Config.USE_SAM_HQ else "standard",
                "val_loss": val_loss_avg,
                "val_dice": val_dice_avg,
                "mask_decoder_state_dict": mask_decoder.state_dict(),
                "prompt_encoder_state_dict": prompt_encoder.state_dict(),
                "history": history,
            }, best_path)
            print(f"    -> Melhor modelo salvo: {best_path.name} (Dice: {val_dice_avg:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= Config.EARLY_STOPPING_PATIENCE:
                print(f"    -> Early stopping: {epochs_without_improvement} epocas sem melhoria")
                break

    # Salvar modelo final
    final_path = Config.MODELS_DIR / f"sam_finetuned_{feature}_final.pth"
    torch.save({
        "feature": feature,
        "epoch": epochs,
        "sam_variant": "hq" if Config.USE_SAM_HQ else "standard",
        "val_loss": val_loss_avg,
        "val_dice": val_dice_avg,
        "mask_decoder_state_dict": mask_decoder.state_dict(),
        "prompt_encoder_state_dict": prompt_encoder.state_dict(),
        "history": history,
    }, final_path)

    # Salvar histórico
    history_path = Config.MODELS_DIR / f"training_history_{feature}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Treinamento concluido para '{feature}'!")
    print(f"  Melhor Val Dice: {best_val_dice:.4f}")
    print(f"  Modelos salvos em: {Config.MODELS_DIR}")
    print(f"  Historico salvo em: {history_path}")


def train_feature_onthefly(feature: str, epochs: int, batch_size: int, lr: float,
                           use_augment: bool = True, use_lora: bool = False,
                           pretrained_path: str = None):
    """Treina o SAM com encoding on-the-fly (augmentacao + LoRA opcional).

    Diferente de train_feature(), o encoder roda durante o treino em vez de
    usar embeddings pre-computados. Isso permite:
    - Data augmentation (transformacoes na imagem antes do encoder)
    - LoRA no encoder (fine-tuning parcial das projecoes QKV)

    Estrategia de memoria (GPU <= 4GB):
    - Encoder em float16 dentro de torch.no_grad() (sem LoRA)
    - Encoder em float16 com grad apenas nos params LoRA (com LoRA)
    - batch_size=1 obrigatorio
    - Limpeza de VRAM a cada amostra

    Args:
        feature: Nome da feicao.
        epochs: Numero de epocas.
        batch_size: Tamanho do batch (forcado a 1 internamente).
        lr: Learning rate para decoder/prompt_encoder.
        use_augment: Se True, aplica augmentacao no treino.
        use_lora: Se True, injeta LoRA no encoder e treina junto.
    """
    # Forcar batch_size=1 para caber em 4GB VRAM
    if batch_size > 1:
        print(f"  [AVISO] batch_size={batch_size} forcado para 1 (modo on-the-fly, VRAM limitada)")
        batch_size = 1

    print(f"\n{'='*60}")
    print(f"FINE-TUNING SAM (on-the-fly) - Feicao: {feature.upper()}")
    print(f"{'='*60}")
    modo = []
    if use_augment:
        modo.append("augmentacao")
    if use_lora:
        modo.append("LoRA")
    print(f"  Modo: {' + '.join(modo) if modo else 'on-the-fly basico'}")

    # Coletar dados
    tile_paths, mask_paths = collect_pairs(feature)

    if len(tile_paths) == 0:
        print(f"[ERRO] Nenhuma anotacao encontrada para '{feature}'.")
        for year in Config.YEARS:
            d = Config.MASKS_DIR / str(year) / "annotations" / feature
            print(f"  {d} (existe: {d.exists()})")
        return

    print(f"  Amostras encontradas: {len(tile_paths)}")

    # Split treino/validacao
    train_tiles, train_masks, val_tiles, val_masks = split_dataset(
        tile_paths, mask_paths, Config.TRAIN_VAL_SPLIT
    )
    print(f"  Treino: {len(train_tiles)} | Validacao: {len(val_tiles)}")

    # Criar datasets (on-the-fly, sem embeddings pre-computados)
    train_transform = get_train_transform() if use_augment else None
    use_cp = Config.USE_COPY_PASTE and use_augment
    train_dataset = GlacierOnTheFlyDataset(
        train_tiles, train_masks, transform=train_transform,
        use_copy_paste=use_cp, copy_paste_prob=Config.COPY_PASTE_PROB,
    )
    val_dataset = GlacierOnTheFlyDataset(val_tiles, val_masks, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Carregar SAM completo (encoder + decoder + prompt encoder)
    sam_variant = "SAM-HQ" if Config.USE_SAM_HQ else "SAM"
    print(f"  Carregando {sam_variant} completo ({Config.MODEL_TYPE})...")
    sam = sam_model_registry[Config.MODEL_TYPE](checkpoint=str(Config.SAM_CHECKPOINT))
    sam.to(Config.DEVICE)

    # Carregar pesos pre-treinados com satelite (warm-start)
    if pretrained_path:
        print(f"  Carregando pesos pre-treinados de: {pretrained_path}")
        pretrained_ckpt = torch.load(pretrained_path, map_location=Config.DEVICE)
        sam.mask_decoder.load_state_dict(pretrained_ckpt["mask_decoder_state_dict"])
        sam.prompt_encoder.load_state_dict(pretrained_ckpt["prompt_encoder_state_dict"])
        print(f"  ✓ Warm-start aplicado (pre-treinado em satelite)")

    # Encoder em float16 para economizar VRAM
    sam.image_encoder.half()
    sam.image_encoder.eval()

    # Congelar encoder por padrao (todos os params)
    for param in sam.image_encoder.parameters():
        param.requires_grad = False

    # Injetar LoRA se solicitado
    lora_params = []
    if use_lora:
        n_lora = inject_lora(
            sam.image_encoder,
            r=Config.LORA_RANK,
            alpha=Config.LORA_ALPHA,
            dropout=Config.LORA_DROPOUT,
        )
        # Coletar params LoRA (apenas estes tem requires_grad=True no encoder)
        lora_params = [p for p in sam.image_encoder.parameters() if p.requires_grad]
        n_lora_params = sum(p.numel() for p in lora_params)
        print(f"  LoRA injetado: {n_lora} camadas, {n_lora_params:,} params ({n_lora_params/1e6:.2f}M)")
        # LoRA params em float32 para estabilidade
        for p in lora_params:
            p.data = p.data.float()

    # Decoder e prompt encoder treinaveis (float32)
    sam.mask_decoder.train()
    sam.prompt_encoder.train()
    decoder_params = list(sam.mask_decoder.parameters()) + list(sam.prompt_encoder.parameters())
    for param in decoder_params:
        param.requires_grad = True

    total_decoder = sum(p.numel() for p in decoder_params)
    total_lora = sum(p.numel() for p in lora_params)
    total_trainable = total_decoder + total_lora
    print(f"  Params treinaveis: decoder={total_decoder:,} ({total_decoder/1e6:.1f}M)"
          f" + LoRA={total_lora:,} ({total_lora/1e6:.2f}M)"
          f" = {total_trainable:,} ({total_trainable/1e6:.1f}M)")

    # Optimizer com LR diferenciada
    param_groups = [{"params": decoder_params, "lr": lr}]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": Config.LORA_ENCODER_LR})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    criterion = ImprovedLoss(feature=feature)

    # Scheduler: warmup linear + cosine annealing
    warmup_epochs = Config.WARMUP_EPOCHS
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_dice = 0.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "val_dice": []}

    # Positional encoding (constante)
    image_pe = sam.prompt_encoder.get_dense_pe()

    # Estrategias de prompt
    strategies = ["random", "center", "bbox"]

    for epoch in range(1, epochs + 1):
        # --- TREINO ---
        sam.mask_decoder.train()
        sam.prompt_encoder.train()
        if use_lora:
            sam.image_encoder.train()  # LoRA precisa de train mode (dropout)

        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"  Epoca {epoch:02d}/{epochs} [treino]")
        for batch_idx, (images, masks_gt) in enumerate(pbar):
            images = images.to(Config.DEVICE)        # (B, 3, 1024, 1024)
            masks_gt = masks_gt.to(Config.DEVICE)    # (B, 1, 256, 256)

            strategy = strategies[batch_idx % len(strategies)]

            optimizer.zero_grad()
            batch_loss_sum = 0.0
            valid_samples = 0
            bs = images.size(0)

            for i in range(bs):
                img_i = images[i:i+1].half()  # float16 para encoder
                mask_i = masks_gt[i:i+1]

                # Encoder (float16, grad apenas para LoRA se ativo)
                if use_lora:
                    # LoRA params precisam de gradiente
                    if Config.USE_SAM_HQ:
                        emb_i, interm_embs = sam.image_encoder(img_i)
                        emb_i = emb_i.float()
                        interm_embs = [e.float() for e in interm_embs]
                    else:
                        emb_i = sam.image_encoder(img_i).float()
                else:
                    with torch.no_grad():
                        if Config.USE_SAM_HQ:
                            emb_i, interm_embs = sam.image_encoder(img_i)
                            emb_i = emb_i.float()
                            interm_embs = [e.float() for e in interm_embs]
                        else:
                            emb_i = sam.image_encoder(img_i).float()

                del img_i

                # Gerar prompts
                pts, lbls, boxes = generate_prompts(mask_i, Config.DEVICE, strategy)

                if boxes is not None:
                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=None, boxes=boxes, masks=None,
                    )
                else:
                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=(pts, lbls), boxes=None, masks=None,
                    )

                # Decoder
                decoder_kwargs = dict(
                    image_embeddings=emb_i,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                if Config.USE_SAM_HQ:
                    decoder_kwargs["hq_token_only"] = True
                    decoder_kwargs["interm_embeddings"] = interm_embs

                low_res_mask, _ = sam.mask_decoder(**decoder_kwargs)

                loss_i = criterion(low_res_mask, mask_i)
                if not torch.isfinite(loss_i):
                    del emb_i, low_res_mask
                    if Config.USE_SAM_HQ:
                        del interm_embs
                    torch.cuda.empty_cache()
                    continue

                (loss_i / bs).backward()
                batch_loss_sum += loss_i.item()
                valid_samples += 1

                del emb_i, low_res_mask
                if Config.USE_SAM_HQ:
                    del interm_embs
                torch.cuda.empty_cache()

            if valid_samples > 0:
                # Gradient clipping para estabilidade (sempre, nao apenas LoRA)
                all_trainable = decoder_params + lora_params
                torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=1.0)
                optimizer.step()

            train_loss_sum += batch_loss_sum
            train_count += valid_samples
            mean_batch_loss = batch_loss_sum / max(valid_samples, 1)
            pbar.set_postfix(loss=f"{mean_batch_loss:.4f}", prompt=strategy)

        train_loss_avg = train_loss_sum / max(train_count, 1)

        # --- VALIDACAO ---
        sam.mask_decoder.eval()
        sam.prompt_encoder.eval()
        sam.image_encoder.eval()
        val_loss_sum = 0.0
        val_loss_count = 0
        val_dice_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for images, masks_gt in val_loader:
                images = images.to(Config.DEVICE)
                masks_gt = masks_gt.to(Config.DEVICE)

                bs = images.size(0)
                for i in range(bs):
                    img_i = images[i:i+1].half()
                    mask_i = masks_gt[i:i+1]

                    if Config.USE_SAM_HQ:
                        emb_i, interm_embs = sam.image_encoder(img_i)
                        emb_i = emb_i.float()
                        interm_embs = [e.float() for e in interm_embs]
                    else:
                        emb_i = sam.image_encoder(img_i).float()

                    del img_i

                    # Validacao usa centro (deterministico)
                    pts, lbls, _ = generate_prompts(mask_i, Config.DEVICE, "center")

                    sparse_emb, dense_emb = sam.prompt_encoder(
                        points=(pts, lbls), boxes=None, masks=None,
                    )

                    decoder_kwargs = dict(
                        image_embeddings=emb_i,
                        image_pe=image_pe,
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False,
                    )
                    if Config.USE_SAM_HQ:
                        decoder_kwargs["hq_token_only"] = True
                        decoder_kwargs["interm_embeddings"] = interm_embs

                    low_res_mask, _ = sam.mask_decoder(**decoder_kwargs)

                    loss = criterion(low_res_mask, mask_i)
                    if torch.isfinite(loss):
                        val_loss_sum += loss.item()
                        val_loss_count += 1

                    pred_binary = (torch.sigmoid(low_res_mask) > 0.5).float()
                    intersection = (pred_binary * mask_i).sum()
                    union_val = pred_binary.sum() + mask_i.sum()
                    dice = (2.0 * intersection + 1.0) / (union_val + 1.0)
                    val_dice_sum += dice.item()

                    val_count += 1

                    del emb_i, low_res_mask
                    if Config.USE_SAM_HQ:
                        del interm_embs
                    torch.cuda.empty_cache()

        val_loss_avg = val_loss_sum / max(val_loss_count, 1)
        val_dice_avg = val_dice_sum / max(val_count, 1)

        scheduler.step()

        history["train_loss"].append(train_loss_avg)
        history["val_loss"].append(val_loss_avg)
        history["val_dice"].append(val_dice_avg)

        current_lr = scheduler.get_last_lr()[0]
        print(f"  Epoca {epoch:02d}/{epochs} | "
              f"Train Loss: {train_loss_avg:.4f} | "
              f"Val Loss: {val_loss_avg:.4f} | "
              f"Val Dice: {val_dice_avg:.4f} | "
              f"LR: {current_lr:.2e}")

        # Salvar melhor modelo (por val_dice, mais robusto que val_loss)
        if val_dice_avg > best_val_dice + Config.EARLY_STOPPING_MIN_DELTA:
            best_val_dice = val_dice_avg
            epochs_without_improvement = 0
            best_path = Config.MODELS_DIR / f"sam_finetuned_{feature}_best.pth"
            save_dict = {
                "feature": feature,
                "epoch": epoch,
                "sam_variant": "hq" if Config.USE_SAM_HQ else "standard",
                "training_mode": "onthefly",
                "use_augmentation": use_augment,
                "use_lora": use_lora,
                "val_loss": val_loss_avg,
                "val_dice": val_dice_avg,
                "mask_decoder_state_dict": sam.mask_decoder.state_dict(),
                "prompt_encoder_state_dict": sam.prompt_encoder.state_dict(),
                "history": history,
            }
            if use_lora:
                # Salvar apenas os pesos LoRA do encoder (nao o encoder inteiro)
                lora_state = {
                    k: v for k, v in sam.image_encoder.state_dict().items()
                    if "lora_" in k
                }
                save_dict["lora_state_dict"] = lora_state
                save_dict["lora_config"] = {
                    "rank": Config.LORA_RANK,
                    "alpha": Config.LORA_ALPHA,
                    "dropout": Config.LORA_DROPOUT,
                }
            torch.save(save_dict, best_path)
            print(f"    -> Melhor modelo salvo: {best_path.name} (Dice: {val_dice_avg:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= Config.EARLY_STOPPING_PATIENCE:
                print(f"    -> Early stopping: {epochs_without_improvement} epocas sem melhoria")
                break

    # Salvar modelo final
    final_path = Config.MODELS_DIR / f"sam_finetuned_{feature}_final.pth"
    save_dict = {
        "feature": feature,
        "epoch": epochs,
        "sam_variant": "hq" if Config.USE_SAM_HQ else "standard",
        "training_mode": "onthefly",
        "use_augmentation": use_augment,
        "use_lora": use_lora,
        "val_loss": val_loss_avg,
        "val_dice": val_dice_avg,
        "mask_decoder_state_dict": sam.mask_decoder.state_dict(),
        "prompt_encoder_state_dict": sam.prompt_encoder.state_dict(),
        "history": history,
    }
    if use_lora:
        lora_state = {
            k: v for k, v in sam.image_encoder.state_dict().items()
            if "lora_" in k
        }
        save_dict["lora_state_dict"] = lora_state
        save_dict["lora_config"] = {
            "rank": Config.LORA_RANK,
            "alpha": Config.LORA_ALPHA,
            "dropout": Config.LORA_DROPOUT,
        }
    torch.save(save_dict, final_path)

    # Salvar historico
    history_path = Config.MODELS_DIR / f"training_history_{feature}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Liberar GPU
    del sam
    torch.cuda.empty_cache()

    print(f"\n  Treinamento concluido para '{feature}'!")
    print(f"  Melhor Val Dice: {best_val_dice:.4f}")
    print(f"  Modelos salvos em: {Config.MODELS_DIR}")
    print(f"  Historico salvo em: {history_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning do SAM para feições supraglaciais"
    )
    parser.add_argument(
        "--feature", type=str, default=None,
        choices=list(Config.FEATURES.keys()),
        help="Feição específica para treinar. Se omitido, treina todas."
    )
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    parser.add_argument(
        "--augment", action="store_true", default=Config.USE_AUGMENTATION,
        help="Ativar data augmentation (encoding on-the-fly, requer albumentations)"
    )
    parser.add_argument(
        "--lora", action="store_true", default=Config.USE_LORA,
        help="Ativar LoRA no encoder (requer --augment, encoding on-the-fly)"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Caminho para checkpoint de pre-treinamento satelite (03a_pretrain_satellite.py). "
             "Carrega pesos do decoder/prompt_encoder como warm-start antes do fine-tuning."
    )
    args = parser.parse_args()

    # LoRA implica augmentacao (ambos precisam de on-the-fly)
    if args.lora and not args.augment:
        print("[AVISO] --lora implica --augment (encoding on-the-fly). Ativando augmentacao.")
        args.augment = True

    use_onthefly = args.augment or args.lora

    print("=" * 60)
    print("FASE 3 - FINE-TUNING DO SAM")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Backend: {SAM_BACKEND}")
    print(f"Epocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Augmentacao: {'SIM' if args.augment else 'NAO'}")
    print(f"LoRA: {'SIM (r={}, alpha={}, encoder_lr={})'.format(Config.LORA_RANK, Config.LORA_ALPHA, Config.LORA_ENCODER_LR) if args.lora else 'NAO'}")
    if args.pretrained:
        print(f"Pre-treinamento: {args.pretrained}")
    if use_onthefly:
        print(f"Modo: encoding on-the-fly (encoder fp16 + decoder fp32)")
    else:
        print(f"Modo: embeddings pre-computados (fp16) + decoder-only")

    start = time.time()

    if args.feature:
        features = [args.feature]
    else:
        features = list(Config.FEATURES.keys())

    for feature in features:
        if use_onthefly:
            train_feature_onthefly(
                feature, args.epochs, args.batch_size, args.lr,
                use_augment=args.augment, use_lora=args.lora,
                pretrained_path=args.pretrained,
            )
        else:
            train_feature(feature, args.epochs, args.batch_size, args.lr,
                          pretrained_path=args.pretrained)

    elapsed = time.time() - start
    print(f"\nTempo total: {elapsed/60:.1f} minutos")


if __name__ == "__main__":
    main()
