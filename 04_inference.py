"""
04_inference.py - Inferência em larga escala com SAM fine-tuned

Projeto: LACRIO IC - Extração de Feições Supraglaciais
Fase 4: Aplicar modelo fine-tuned em todos os ~22k tiles

Estratégia de memória (GPU <= 4GB):
  - Encoder roda em float16, 1 tile por vez
  - Decoder roda em float32 (leve, ~16MB)
  - Libera VRAM após cada tile
  Consumo estimado: ~3 GB VRAM pico

Uso:
    python 04_inference.py                              # Todas as feições, todos os anos
    python 04_inference.py --feature lakes --year 2016  # Feição e ano específicos
    python 04_inference.py --feature lakes --year 2016 --annotated-only  # Apenas tiles com GT
    python 04_inference.py --threshold 0.5              # Ajustar threshold
    python 04_inference.py --feature lakes --year 2016 --combine-mode mean --pred-iou-threshold 0.8
    python 04_inference.py --feature lakes --year 2016 --lakes-preset precision
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
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
    SAM_BACKEND = "segment_anything"

from shadow_utils import (
    precompute_year_shadows,
    get_shadow_mask_for_tile,
    filter_by_texture,
)

import torch.nn.functional as F


def load_finetuned_sam(feature: str):
    """Carrega o SAM com pesos fine-tuned, encoder em float16.

    Mantém o encoder em float16 para economizar VRAM.
    Decoder e prompt encoder ficam em float32 (pesos fine-tuned).
    Se o checkpoint contém pesos LoRA, injeta e carrega no encoder.

    Args:
        feature: Nome da feição (lakes, crevasses, channels).

    Returns:
        sam: Modelo SAM com decoder fine-tuned.
    """
    checkpoint_path = Config.MODELS_DIR / f"sam_finetuned_{feature}_best.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Modelo fine-tuned não encontrado: {checkpoint_path}\n"
            f"Execute primeiro: python 03_finetune_sam.py --feature {feature}"
        )

    # Carregar SAM base
    sam = sam_model_registry[Config.MODEL_TYPE](checkpoint=str(Config.SAM_CHECKPOINT))

    # Carregar pesos fine-tuned
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Verificar compatibilidade SAM variant
    saved_variant = checkpoint.get("sam_variant", "standard")
    current_variant = "hq" if Config.USE_SAM_HQ else "standard"
    if saved_variant != current_variant:
        print(f"  [AVISO] Checkpoint treinado com SAM {saved_variant}, "
              f"mas usando SAM {current_variant}. Pode haver incompatibilidade!")

    sam.mask_decoder.load_state_dict(checkpoint["mask_decoder_state_dict"])
    sam.prompt_encoder.load_state_dict(checkpoint["prompt_encoder_state_dict"])

    # Carregar LoRA se presente no checkpoint
    if "lora_state_dict" in checkpoint and checkpoint.get("use_lora", False):
        lora_cfg = checkpoint.get("lora_config", {})

        # LoRALinear inline (evita dependencia circular com 03_finetune_sam)
        class _LoRALinear(nn.Module):
            def __init__(self, original_layer, r=4, alpha=16):
                super().__init__()
                self.original = original_layer
                self.scale = alpha / r
                self.lora_A = nn.Linear(original_layer.in_features, r, bias=False)
                self.lora_B = nn.Linear(r, original_layer.out_features, bias=False)
                self.lora_dropout = nn.Dropout(0.0)  # sem dropout na inferencia
                self.original.weight.requires_grad = False
                if self.original.bias is not None:
                    self.original.bias.requires_grad = False
            def forward(self, x):
                return self.original(x) + self.lora_B(self.lora_A(x)) * self.scale

        r = lora_cfg.get("rank", Config.LORA_RANK)
        alpha = lora_cfg.get("alpha", Config.LORA_ALPHA)
        n_lora = 0
        for block in sam.image_encoder.blocks:
            if hasattr(block.attn, "qkv") and isinstance(block.attn.qkv, nn.Linear):
                block.attn.qkv = _LoRALinear(block.attn.qkv, r=r, alpha=alpha)
                n_lora += 1

        # Carregar pesos LoRA
        lora_state = checkpoint["lora_state_dict"]
        encoder_state = sam.image_encoder.state_dict()
        encoder_state.update(lora_state)
        sam.image_encoder.load_state_dict(encoder_state)
        print(f"  LoRA carregado: {n_lora} camadas (r={r}, alpha={alpha})")

    # Encoder em float16 para economizar VRAM
    sam.image_encoder.to(Config.DEVICE).half()
    sam.image_encoder.eval()

    # Decoder e prompt encoder em float32
    sam.mask_decoder.to(Config.DEVICE)
    sam.mask_decoder.eval()
    sam.prompt_encoder.to(Config.DEVICE)
    sam.prompt_encoder.eval()

    sam_variant = "SAM-HQ" if Config.USE_SAM_HQ else "SAM"
    training_mode = checkpoint.get("training_mode", "embeddings")
    use_aug = checkpoint.get("use_augmentation", False)
    use_lora = checkpoint.get("use_lora", False)
    print(f"  Modelo carregado: {checkpoint_path.name} ({sam_variant})")
    print(f"  Treinado na epoca {checkpoint['epoch']} | "
          f"Val Dice: {checkpoint.get('val_dice', 'N/A'):.4f}")
    print(f"  Modo treino: {training_mode} | Aug: {use_aug} | LoRA: {use_lora}")
    print(f"  Encoder: float16 | Decoder: float32")

    return sam


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


def compute_slope_map(year):
    """Computa mapa de slope em graus a partir do DEM de um ano.

    Lagos supraglaciais nao existem em encostas ingremes (a agua escorre).
    Deteccoes em slopes > threshold sao quase certamente FP.

    Args:
        year: Ano do DEM.

    Returns:
        slope_deg: Array (H, W) float32 com slope em graus, ou None se DEM indisponivel.
        dem_transform: Transform rasterio do DEM.
    """
    import rasterio

    try:
        dem_path = Config.get_dem_path(year)
    except ValueError:
        return None, None

    if not dem_path.exists():
        return None, None

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        dem_transform = src.transform
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)

    # Substituir NoData por NaN
    dem[dem < -9000] = np.nan

    # Gradiente
    dy, dx = np.gradient(dem, res_y, res_x)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    slope_deg = np.nan_to_num(slope_deg, nan=0.0)

    return slope_deg, dem_transform


def get_slope_for_tile(tile_info, slope_map, dem_transform):
    """Extrai slope medio para a regiao de um tile.

    Args:
        tile_info: Dict com 'col_off', 'row_off', 'width', 'height' do tile.
        slope_map: Array (H, W) de slope em graus.
        dem_transform: Transform rasterio do DEM.

    Returns:
        mean_slope: Slope medio em graus na regiao do tile.
    """
    import rasterio
    from rasterio.windows import Window

    # Ler info geometrica do tile (em coordenadas do mosaico)
    col_off = tile_info.get("col_off", 0)
    row_off = tile_info.get("row_off", 0)
    width = tile_info.get("width", Config.TILE_SIZE)
    height = tile_info.get("height", Config.TILE_SIZE)

    # Converter coordenadas do mosaico para o DEM (resolucoes diferentes)
    # Mosaico: 5.4 cm/pixel, DEM: 22 cm/pixel => fator ~4x
    mosaic_path = Config.get_mosaic_path(int(tile_info.get("year", 2016)))
    try:
        with rasterio.open(mosaic_path) as mosaic_src:
            # Pixel do mosaico -> coordenada geografica
            x_geo, y_geo = mosaic_src.xy(row_off + height // 2, col_off + width // 2)

        # Coordenada geografica -> pixel do DEM
        inv_transform = ~dem_transform
        dem_col, dem_row = inv_transform * (x_geo, y_geo)
        dem_col, dem_row = int(dem_col), int(dem_row)

        # Regiao correspondente no DEM (~4x menor)
        dem_hw = max(1, width // 4)
        r1 = max(0, dem_row - dem_hw // 2)
        r2 = min(slope_map.shape[0], dem_row + dem_hw // 2)
        c1 = max(0, dem_col - dem_hw // 2)
        c2 = min(slope_map.shape[1], dem_col + dem_hw // 2)

        if r1 >= r2 or c1 >= c2:
            return 0.0

        return float(slope_map[r1:r2, c1:c2].mean())
    except Exception:
        return 0.0


def predict_tile_proba(sam, image, pred_iou_threshold=0.5, combine_mode="vote",
                       vote_threshold=3):
    """Gera mapa de probabilidade (float32) para um tile, sem binarizar.

    Retorna probabilidades em vez de mascara binaria, para permitir
    TTA (media de probabilidades) e refinamento 2-pass.

    Args:
        sam: Modelo SAM fine-tuned.
        image: Imagem RGB (H, W, 3) uint8.
        pred_iou_threshold: Limiar minimo de IoU predito por prompt.
        combine_mode: Estrategia de combinacao dos prompts.
        vote_threshold: Minimo de pontos concordantes para ativar pixel (modo vote).

    Returns:
        prob_map: Mapa de probabilidade (256, 256) float32 [0, 1].
        image_embedding: Embedding da imagem (para reuso no 2-pass).
        interm_embeddings: Features intermediarias SAM-HQ (ou None).
    """
    image_resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = normalize_image(image_tensor)
    image_tensor = image_tensor.to(Config.DEVICE).half()

    with torch.no_grad():
        if Config.USE_SAM_HQ:
            image_embedding, interm_embeddings = sam.image_encoder(image_tensor)
            image_embedding = image_embedding.float()
            interm_embeddings = [e.float() for e in interm_embeddings]
        else:
            image_embedding = sam.image_encoder(image_tensor)
            image_embedding = image_embedding.float()
            interm_embeddings = None

        del image_tensor
        torch.cuda.empty_cache()

        # Grade densa de pontos (8x8 = 64 pontos)
        stride = 128
        coords = np.arange(stride // 2, 1024, stride)
        xx, yy = np.meshgrid(coords, coords)
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)

        max_mask = np.zeros((256, 256), dtype=np.float32)
        sum_mask = np.zeros((256, 256), dtype=np.float32)
        vote_mask = np.zeros((256, 256), dtype=np.float32)
        valid_points = 0

        image_pe = sam.prompt_encoder.get_dense_pe()
        label_tensor = torch.ones(1, 1, dtype=torch.int, device=Config.DEVICE)

        for point in grid_points:
            point_tensor = torch.from_numpy(point[None, None, :]).to(Config.DEVICE)

            sparse_emb, dense_emb = sam.prompt_encoder(
                points=(point_tensor, label_tensor),
                boxes=None,
                masks=None,
            )

            decoder_kwargs = dict(
                image_embeddings=image_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            if Config.USE_SAM_HQ:
                decoder_kwargs["hq_token_only"] = True
                decoder_kwargs["interm_embeddings"] = interm_embeddings

            low_res_mask, iou_pred = sam.mask_decoder(**decoder_kwargs)

            pred_iou = iou_pred[0, 0].item()
            if pred_iou < pred_iou_threshold:
                continue

            valid_points += 1
            mask_prob = torch.sigmoid(low_res_mask[0, 0]).cpu().numpy()
            max_mask = np.maximum(max_mask, mask_prob)
            sum_mask += mask_prob
            vote_mask += (mask_prob > 0.5).astype(np.float32)

    if valid_points == 0:
        prob_map = np.zeros((256, 256), dtype=np.float32)
    elif combine_mode == "mean":
        prob_map = sum_mask / valid_points
    elif combine_mode == "vote":
        # Pixel ativado apenas se >= vote_threshold pontos concordam
        prob_map = np.where(vote_mask >= vote_threshold,
                            vote_mask / valid_points, 0.0).astype(np.float32)
    else:
        prob_map = max_mask

    return prob_map, image_embedding, interm_embeddings


def apply_mask_refinement(sam, prob_map, image_embedding, interm_embeddings):
    """Refinamento 2-pass: usa mascara do 1o pass como prompt para o 2o.

    O SAM aceita uma mascara como prompt adicional. Passando a predicao
    inicial como mask prompt, o decoder pode refinar bordas e corrigir erros.

    Args:
        sam: Modelo SAM fine-tuned.
        prob_map: Mapa de probabilidade do 1o pass (256, 256).
        image_embedding: Embedding da imagem.
        interm_embeddings: Features intermediarias SAM-HQ (ou None).

    Returns:
        refined_prob: Mapa de probabilidade refinado (256, 256).
    """
    with torch.no_grad():
        # Mascara como prompt (256x256 -> 256x256, ja no tamanho certo)
        mask_input = torch.from_numpy(prob_map).unsqueeze(0).unsqueeze(0).float()
        mask_input = mask_input.to(Config.DEVICE)

        # Ponto no centroide da mascara como prompt adicional
        binary = (prob_map > 0.5).astype(np.uint8)
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return prob_map

        cx = float(xs.mean()) * (1024.0 / 256.0)
        cy = float(ys.mean()) * (1024.0 / 256.0)
        point_tensor = torch.tensor([[[cx, cy]]], dtype=torch.float32, device=Config.DEVICE)
        label_tensor = torch.ones(1, 1, dtype=torch.int, device=Config.DEVICE)

        image_pe = sam.prompt_encoder.get_dense_pe()

        sparse_emb, dense_emb = sam.prompt_encoder(
            points=(point_tensor, label_tensor),
            boxes=None,
            masks=mask_input,
        )

        decoder_kwargs = dict(
            image_embeddings=image_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        if Config.USE_SAM_HQ and interm_embeddings is not None:
            decoder_kwargs["hq_token_only"] = True
            decoder_kwargs["interm_embeddings"] = interm_embeddings

        low_res_mask, _ = sam.mask_decoder(**decoder_kwargs)
        refined_prob = torch.sigmoid(low_res_mask[0, 0]).cpu().numpy()

    return refined_prob


def predict_with_tta(sam, image, threshold=0.3, pred_iou_threshold=0.5,
                     combine_mode="vote", use_refinement=False,
                     vote_threshold=3):
    """Predicao com Test-Time Augmentation (TTA).

    Roda predicao na imagem original + versoes augmentadas (flips, rotacao),
    desfaz as transformacoes, e faz media das probabilidades.
    Melhora IoU em +2-4% ao custo de 4x mais tempo.

    Args:
        sam: Modelo SAM fine-tuned.
        image: Imagem RGB (H, W, 3) uint8.
        threshold: Limiar de confianca para binarizacao.
        pred_iou_threshold: Limiar minimo de IoU predito.
        combine_mode: Estrategia de combinacao dos prompts.
        use_refinement: Se True, aplica 2-pass refinement apos TTA.

    Returns:
        mask: Mascara binaria (H, W) uint8 (0 ou 255).
    """
    h, w = image.shape[:2]

    transforms = [
        ("original", lambda x: x, lambda x: x),
        ("hflip", lambda x: np.fliplr(x).copy(), lambda x: np.fliplr(x).copy()),
        ("vflip", lambda x: np.flipud(x).copy(), lambda x: np.flipud(x).copy()),
        ("rot180", lambda x: np.rot90(x, 2).copy(), lambda x: np.rot90(x, -2).copy()),
    ]

    # Filtrar transforms conforme config
    active_names = set(Config.TTA_TRANSFORMS)
    transforms = [(name, fwd, inv) for name, fwd, inv in transforms if name in active_names]

    probs = []
    last_embedding = None
    last_interm = None

    for name, fwd, inv in transforms:
        aug_image = fwd(image)
        prob_map, emb, interm = predict_tile_proba(
            sam, aug_image, pred_iou_threshold, combine_mode,
            vote_threshold=vote_threshold,
        )
        # Desfazer transformacao na probabilidade
        prob_back = inv(prob_map)
        probs.append(prob_back)

        # Guardar embedding da imagem original para refinement
        if name == "original":
            last_embedding = emb
            last_interm = interm
        else:
            del emb
            if interm is not None:
                del interm
            torch.cuda.empty_cache()

    # Media das probabilidades
    avg_prob = np.mean(probs, axis=0)

    # Refinamento 2-pass (usar embedding da imagem original)
    if use_refinement and last_embedding is not None:
        avg_prob = apply_mask_refinement(sam, avg_prob, last_embedding, last_interm)

    # Limpar
    if last_embedding is not None:
        del last_embedding
    if last_interm is not None:
        del last_interm
    torch.cuda.empty_cache()

    # Binarizar e redimensionar
    binary_mask = (avg_prob > threshold).astype(np.uint8) * 255
    binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return binary_mask


def predict_tile(sam, image, threshold=0.3, pred_iou_threshold=0.5,
                  combine_mode="vote", vote_threshold=3):
    """Gera máscara de segmentação para um tile.

    Estratégia: grade densa de pontos (8x8) + filtra por IoU predito.
    Pontos com IoU baixo (modelo incerto) são descartados.
    O mapa final pode combinar prompts por:
    - "max": pega o maior score por pixel (alta sensibilidade)
    - "mean": média dos scores dos prompts válidos (mais conservador)
    - "vote": pixel ativo se >= vote_threshold pontos concordam (recomendado)

    Args:
        sam: Modelo SAM fine-tuned.
        image: Imagem RGB (H, W, 3) uint8.
        threshold: Limiar de confiança para binarização.
        pred_iou_threshold: Limiar mínimo de IoU predito por prompt.
        combine_mode: Estratégia de combinação dos prompts (max/mean/vote).
        vote_threshold: Minimo de pontos concordantes para ativar pixel (modo vote).

    Returns:
        mask: Máscara binária (H, W) uint8 (0 ou 255).
    """
    h, w = image.shape[:2]

    # Redimensionar para entrada do SAM (1024x1024)
    image_resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0)

    # Normalizar com stats SAM/ImageNet
    image_tensor = normalize_image(image_tensor)
    image_tensor = image_tensor.to(Config.DEVICE).half()  # float16

    with torch.no_grad():
        # Encoder (float16)
        if Config.USE_SAM_HQ:
            image_embedding, interm_embeddings = sam.image_encoder(image_tensor)
            image_embedding = image_embedding.float()
            interm_embeddings = [e.float() for e in interm_embeddings]
        else:
            image_embedding = sam.image_encoder(image_tensor)
            image_embedding = image_embedding.float()

        del image_tensor
        torch.cuda.empty_cache()

        # Grade densa de pontos (8x8 = 64 pontos)
        stride = 128
        coords = np.arange(stride // 2, 1024, stride)
        xx, yy = np.meshgrid(coords, coords)
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)

        max_mask = np.zeros((256, 256), dtype=np.float32)
        sum_mask = np.zeros((256, 256), dtype=np.float32)
        vote_mask = np.zeros((256, 256), dtype=np.float32)
        valid_points = 0

        image_pe = sam.prompt_encoder.get_dense_pe()

        label_tensor = torch.ones(1, 1, dtype=torch.int, device=Config.DEVICE)

        for point in grid_points:
            # Evita criação lenta de tensor a partir de lista de np.ndarray
            point_tensor = torch.from_numpy(point[None, None, :]).to(Config.DEVICE)

            sparse_emb, dense_emb = sam.prompt_encoder(
                points=(point_tensor, label_tensor),
                boxes=None,
                masks=None,
            )

            decoder_kwargs = dict(
                image_embeddings=image_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            if Config.USE_SAM_HQ:
                decoder_kwargs["hq_token_only"] = True
                decoder_kwargs["interm_embeddings"] = interm_embeddings

            low_res_mask, iou_pred = sam.mask_decoder(**decoder_kwargs)

            # Descartar predições de baixa confiança
            pred_iou = iou_pred[0, 0].item()
            if pred_iou < pred_iou_threshold:
                continue

            valid_points += 1
            mask_prob = torch.sigmoid(low_res_mask[0, 0]).cpu().numpy()
            max_mask = np.maximum(max_mask, mask_prob)
            sum_mask += mask_prob
            vote_mask += (mask_prob > 0.5).astype(np.float32)

        if Config.USE_SAM_HQ:
            del image_embedding, interm_embeddings
        else:
            del image_embedding
        torch.cuda.empty_cache()

    if valid_points == 0:
        combined_mask = np.zeros((256, 256), dtype=np.float32)
    elif combine_mode == "mean":
        combined_mask = sum_mask / valid_points
    elif combine_mode == "vote":
        combined_mask = np.where(vote_mask >= vote_threshold,
                                 vote_mask / valid_points, 0.0).astype(np.float32)
    else:
        combined_mask = max_mask

    # Binarizar
    binary_mask = (combined_mask > threshold).astype(np.uint8) * 255

    # Redimensionar de volta para tamanho original
    binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return binary_mask


def find_lake_candidates(image, min_area=100, blue_ratio_thresh=1.4):
    """Detector espectral: encontra regioes azuladas/escuras candidatas a lago.

    Assume imagem RGB (R=ch0, G=ch1, B=ch2).
    Usa B/R ratio, agua escura, e NDWI proxy (G-R)/(G+R) para robustez.
    Filtra por textura (variancia local) para separar agua (lisa) de gelo (rugoso).

    Retorna lista de (bbox, area_candidato) e mascara espectral dilatada.
    """
    r = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    b = image[:, :, 2].astype(np.float32)
    brightness = (r + g + b) / 3.0

    # Azul dominante (B/R > threshold)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(r > 10, b / r, 0)
    is_blue = ratio > blue_ratio_thresh

    # NDWI proxy com RGB: (G - R) / (G + R)
    # Agua tem G > R, mas em glaciares quase tudo tem G > R levemente.
    # Usar NDWI apenas combinado com B dominante para evitar FP massivos.
    with np.errstate(divide='ignore', invalid='ignore'):
        ndwi_proxy = np.where((g + r) > 20, (g - r) / (g + r), 0)
    is_ndwi_water = (ndwi_proxy > 0.1) & (b > r)  # NDWI alto + azulado

    # Agua escura (brilho < 90, tom azulado)
    is_dark_water = (brightness < 90) & (b > r) & (brightness > 30)

    # Nao muito claro (neve/gelo)
    not_bright = brightness < 200

    candidate = ((is_blue | is_dark_water | is_ndwi_water) & not_bright).astype(np.uint8) * 255

    # Morfologia: fechar buracos e remover ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)

    # Filtro de textura: agua e lisa (baixa variancia), gelo/sombra e rugoso
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_var = cv2.blur(gray ** 2, (15, 15)) - cv2.blur(gray, (15, 15)) ** 2
    local_var = np.clip(local_var, 0, None)
    # Rejeitar regioes com textura muito alta (>500 = gelo rugoso)
    high_texture = local_var > 500
    candidate[high_texture] = 0

    # Mascara dilatada (margem para SAM refinar bordas)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    candidate_dilated = cv2.dilate(candidate, dilate_kernel)

    # Extrair bboxes dos componentes conectados
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    h, w = image.shape[:2]
    bboxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        # Margem de 15px
        x1 = max(0, x - 15)
        y1 = max(0, y - 15)
        x2 = min(w, x + bw + 15)
        y2 = min(h, y + bh + 15)
        bboxes.append(([x1, y1, x2, y2], area))

    return bboxes, candidate_dilated


def predict_tile_propose_refine(sam, image, threshold=0.3, pred_iou_threshold=0.5,
                                max_area_ratio=5.0):
    """Inferencia 'propor e refinar' para lagos.

    1. Detector espectral encontra candidatos (regioes azuis/escuras)
    2. SAM refina cada candidato com prompt bbox (igual ao treino)
    3. Saida do SAM e clipada pelo bbox do candidato (impede expansao)
    4. Area da mascara e limitada a max_area_ratio * area do candidato

    Args:
        sam: Modelo SAM fine-tuned.
        image: Imagem RGB (H, W, 3) uint8.
        threshold: Limiar de binarizacao.
        pred_iou_threshold: IoU minimo para aceitar mascara.
        max_area_ratio: Rejeitar mascaras com area > ratio * area_candidato.

    Returns:
        mask: Mascara binaria (H, W) uint8 (0 ou 255).
    """
    h, w = image.shape[:2]
    candidates, spectral_mask = find_lake_candidates(image)

    if len(candidates) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    scale_x = 1024.0 / w
    scale_y = 1024.0 / h

    # Encoder
    image_resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = normalize_image(image_tensor)
    image_tensor = image_tensor.to(Config.DEVICE).half()

    final_mask = np.zeros((h, w), dtype=np.uint8)

    with torch.no_grad():
        if Config.USE_SAM_HQ:
            image_embedding, interm_embeddings = sam.image_encoder(image_tensor)
            image_embedding = image_embedding.float()
            interm_embeddings = [e.float() for e in interm_embeddings]
        else:
            image_embedding = sam.image_encoder(image_tensor).float()
            interm_embeddings = None

        del image_tensor
        torch.cuda.empty_cache()

        image_pe = sam.prompt_encoder.get_dense_pe()

        for bbox, candidate_area in candidates:
            x1, y1, x2, y2 = bbox
            box_1024 = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            box_tensor = torch.tensor([box_1024], dtype=torch.float32,
                                      device=Config.DEVICE).unsqueeze(0)

            sparse_emb, dense_emb = sam.prompt_encoder(
                points=None, boxes=box_tensor, masks=None,
            )

            decoder_kwargs = dict(
                image_embeddings=image_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )
            if Config.USE_SAM_HQ and interm_embeddings is not None:
                decoder_kwargs["hq_token_only"] = True
                decoder_kwargs["interm_embeddings"] = interm_embeddings

            low_res_mask, iou_pred = sam.mask_decoder(**decoder_kwargs)

            pred_iou = iou_pred[0, 0].item()
            if pred_iou < pred_iou_threshold:
                continue

            mask_prob = torch.sigmoid(low_res_mask[0, 0]).cpu().numpy()
            mask_256 = (mask_prob > threshold).astype(np.uint8) * 255
            mask_full = cv2.resize(mask_256, (w, h), interpolation=cv2.INTER_NEAREST)

            # Clipar pelo bbox do candidato (impede SAM de expandir alem)
            bbox_clip = np.zeros_like(mask_full)
            bbox_clip[y1:y2, x1:x2] = 255
            mask_full = cv2.bitwise_and(mask_full, bbox_clip)

            # Filtro de area: rejeitar se mascara >> candidato
            mask_area = (mask_full > 0).sum()
            if candidate_area > 0 and mask_area > max_area_ratio * candidate_area:
                continue

            final_mask = np.maximum(final_mask, mask_full)

        if Config.USE_SAM_HQ:
            del image_embedding, interm_embeddings
        else:
            del image_embedding
        torch.cuda.empty_cache()

    return final_mask


def apply_feature_filter(mask, image, feature,
                         use_feature_filter=True,
                         lake_blue_ratio=1.2,
                         lake_dark_brightness=80.0,
                         lake_max_brightness=200.0,
                         crevasses_max_brightness=150.0,
                         crevasses_min_aspect=3.0,
                         channels_min_aspect=5.0):
    """Aplica filtros específicos por feição para reduzir falsos positivos.

    Critérios definidos no projeto:
    - Lakes: razão azul/vermelho > 1.3
    - Crevasses: baixa luminosidade (< 150) + forma alongada (aspect ratio > 3)
    - Channels: forma linear (aspect ratio > 5)

    Args:
        mask: Máscara binária (H, W) uint8.
        image: Imagem RGB (H, W, 3) uint8.
        feature: Tipo da feição.
        use_feature_filter: Se False, retorna máscara sem pós-filtro de feição.
        lake_blue_ratio: Razão mínima B/R para aceitar lago azulado.
        lake_dark_brightness: Brilho máximo para considerar "água escura".
        lake_max_brightness: Brilho acima do qual rejeita componente (muito claro).
        crevasses_max_brightness: Brilho máximo para aceitar crevasse.
        crevasses_min_aspect: Aspect ratio mínimo para crevasse.
        channels_min_aspect: Aspect ratio mínimo para canal.

    Returns:
        filtered_mask: Máscara filtrada (H, W) uint8.
    """
    if mask.max() == 0:
        return mask

    if not use_feature_filter:
        return mask

    filtered = np.zeros_like(mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    feature_cfg = Config.FEATURES[feature]
    min_area = feature_cfg["min_area"]
    max_area = feature_cfg["max_area"]

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]

        if area < min_area or area > max_area:
            continue

        component_mask = (labels == label_id).astype(np.uint8)

        if feature == "lakes":
            region_pixels = image[component_mask > 0]
            if len(region_pixels) > 0:
                mean_r = region_pixels[:, 0].mean()
                mean_g = region_pixels[:, 1].mean()
                mean_b = region_pixels[:, 2].mean()
                brightness = (mean_r + mean_g + mean_b) / 3

                # Lagos supraglaciais: azul dominante com contraste
                is_blue = mean_r > 0 and mean_b / mean_r > lake_blue_ratio
                # Água escura: escuro mas com tom azulado (não sombra pura)
                is_dark_water = brightness < lake_dark_brightness and mean_b > mean_r
                # Rejeitar regiões muito claras (neve/gelo branco)
                is_too_bright = brightness > lake_max_brightness

                if is_too_bright or not (is_blue or is_dark_water):
                    continue

        elif feature == "crevasses":
            region_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mean_brightness = region_gray[component_mask > 0].mean()
            if mean_brightness >= crevasses_max_brightness:
                continue

            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                rect = cv2.minAreaRect(contours[0])
                w_rect, h_rect = rect[1]
                if min(w_rect, h_rect) > 0:
                    aspect = max(w_rect, h_rect) / min(w_rect, h_rect)
                    if aspect < crevasses_min_aspect:
                        continue

        elif feature == "channels":
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                rect = cv2.minAreaRect(contours[0])
                w_rect, h_rect = rect[1]
                if min(w_rect, h_rect) > 0:
                    aspect = max(w_rect, h_rect) / min(w_rect, h_rect)
                    if aspect < channels_min_aspect:
                        continue

        filtered[component_mask > 0] = 255

    return filtered


def run_inference(feature: str, year: int, threshold: float = 0.5,
                  use_shadow: bool = True, shadow_threshold: int = None,
                  annotated_only: bool = False,
                  pred_iou_threshold: float = 0.5,
                  combine_mode: str = "vote",
                  vote_threshold: int = 3,
                  use_feature_filter: bool = True,
                  lake_blue_ratio: float = 1.2,
                  lake_dark_brightness: float = 80.0,
                  lake_max_brightness: float = 200.0,
                  crevasses_max_brightness: float = 150.0,
                  crevasses_min_aspect: float = 3.0,
                  channels_min_aspect: float = 5.0,
                  use_tta: bool = False,
                  use_refinement: bool = False,
                  use_slope_filter: bool = False,
                  propose_refine: bool = False):
    """Executa inferência em todos os tiles de um ano para uma feição.

    Args:
        feature: Nome da feição.
        year: Ano para processar.
        threshold: Limiar de confiança.
        use_shadow: Se True, aplica subtração de sombra DEM (apenas lakes).
        shadow_threshold: Threshold de hillshade para sombra.
        annotated_only: Se True, processa apenas tiles com ground truth da feição/ano.
        pred_iou_threshold: Limiar mínimo de IoU predito por prompt.
        combine_mode: Estratégia de combinação dos prompts (max/mean/vote).
        use_feature_filter: Se True, aplica filtros de feição (área/cor/geometria).
        lake_blue_ratio: Razão mínima B/R para lagos (filtro espectral).
        lake_dark_brightness: Brilho máximo para "água escura" em lagos.
        lake_max_brightness: Brilho acima do qual rejeita componente em lagos.
        crevasses_max_brightness: Brilho máximo para aceitar crevasse.
        crevasses_min_aspect: Aspect ratio mínimo para crevasse.
        channels_min_aspect: Aspect ratio mínimo para canal.

    Returns:
        stats: Dicionário com estatísticas da inferência.
    """
    print(f"\n--- Inferência: {feature} | Ano: {year} ---")

    tiles_index_path = Config.TILES_DIR / str(year) / "tiles_index.json"
    if not tiles_index_path.exists():
        print(f"  [ERRO] Índice de tiles não encontrado: {tiles_index_path}")
        return None

    with open(tiles_index_path) as f:
        tiles_index = json.load(f)

    tiles = tiles_index["tiles"]
    total_tiles = len(tiles)

    if annotated_only:
        gt_dir = Config.MASKS_DIR / str(year) / "annotations" / feature
        gt_files = sorted(gt_dir.glob(f"tile_*_{feature}.png"))
        annotated_filenames = {
            f"{gt_file.stem.removesuffix(f'_{feature}')}.png" for gt_file in gt_files
        }
        tiles = [t for t in tiles if t.get("filename") in annotated_filenames]
        print(f"  Modo rápido (annotated-only): {len(tiles)}/{total_tiles} tiles")
        if not tiles:
            print(f"  [AVISO] Nenhum tile anotado encontrado em: {gt_dir}")
            return {
                "total": 0, "processed": 0, "with_features": 0,
                "skipped": 0, "shadow_removed": 0,
            }
    else:
        print(f"  Total de tiles: {total_tiles}")

    # Pré-computar máscara de sombra (DEM-based)
    shadow_mask_full = None
    dem_transform = None
    if use_shadow and feature == "lakes":
        shadow_mask_full, dem_transform = precompute_year_shadows(
            year, shadow_threshold
        )

    # Pré-computar slope map (para filtro de lakes em areas ingremes)
    slope_map = None
    slope_transform = None
    if use_slope_filter and feature == "lakes":
        print(f"  Computando mapa de slope do DEM {year}...")
        slope_map, slope_transform = compute_slope_map(year)
        if slope_map is not None:
            print(f"  Slope map: {slope_map.shape}, max={slope_map.max():.1f} graus")
        else:
            print(f"  [AVISO] DEM nao disponivel para {year}, slope filter desativado")

    # Modo de predicao
    if use_tta:
        print(f"  TTA ativado: {Config.TTA_TRANSFORMS} ({len(Config.TTA_TRANSFORMS)}x)")
    if use_refinement:
        print(f"  Refinamento 2-pass ativado")

    # Carregar modelo
    sam = load_finetuned_sam(feature)

    # Diretório de saída
    output_dir = Config.MASKS_DIR / str(year) / feature
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total": len(tiles), "processed": 0, "with_features": 0,
        "skipped": 0, "shadow_removed": 0,
    }

    for tile_info in tqdm(tiles, desc=f"  {feature}/{year}"):
        tile_file = Config.TILES_DIR / str(year) / tile_info["filename"]

        if not tile_file.exists():
            stats["skipped"] += 1
            continue

        image = cv2.imread(str(tile_file))
        if image is None:
            stats["skipped"] += 1
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Predizer máscara
        if propose_refine and feature == "lakes":
            mask = predict_tile_propose_refine(
                sam, image, threshold,
                pred_iou_threshold=pred_iou_threshold,
            )
        elif use_tta:
            mask = predict_with_tta(
                sam, image, threshold,
                pred_iou_threshold=pred_iou_threshold,
                combine_mode=combine_mode,
                use_refinement=use_refinement,
                vote_threshold=vote_threshold,
            )
        else:
            mask = predict_tile(
                sam, image, threshold,
                pred_iou_threshold=pred_iou_threshold,
                combine_mode=combine_mode,
                vote_threshold=vote_threshold,
            )

        # Filtro de slope (DEM) para lakes
        if feature == "lakes" and slope_map is not None and mask.max() > 0:
            mean_slope = get_slope_for_tile(
                {**tile_info, "year": year}, slope_map, slope_transform
            )
            if mean_slope > Config.SLOPE_FILTER_MAX_DEGREES:
                mask[:] = 0
                stats["slope_removed"] = stats.get("slope_removed", 0) + 1

        # Aplicar filtros específicos da feição
        mask = apply_feature_filter(
            mask, image, feature,
            use_feature_filter=use_feature_filter,
            lake_blue_ratio=lake_blue_ratio,
            lake_dark_brightness=lake_dark_brightness,
            lake_max_brightness=lake_max_brightness,
            crevasses_max_brightness=crevasses_max_brightness,
            crevasses_min_aspect=crevasses_min_aspect,
            channels_min_aspect=channels_min_aspect,
        )

        # Subtração de sombra (DEM-based) para lakes
        if feature == "lakes" and shadow_mask_full is not None:
            had_pixels = mask.max() > 0
            tile_shadow = get_shadow_mask_for_tile(
                tile_info, shadow_mask_full, dem_transform
            )
            mask[tile_shadow > 127] = 0
            if had_pixels and mask.max() == 0:
                stats["shadow_removed"] += 1

        # Filtro de textura para lakes (sombras são uniformes)
        if feature == "lakes" and mask.max() > 0:
            mask = filter_by_texture(mask, image)

        stats["processed"] += 1

        tile_id = tile_info["id"]
        output_path = output_dir / f"tile_{tile_id:06d}_{feature}.png"

        if mask.max() > 0:
            cv2.imwrite(str(output_path), mask)
            stats["with_features"] += 1
        elif output_path.exists():
            # Remove predição antiga para evitar validação contaminada por arquivos stale.
            output_path.unlink()

    # Liberar GPU
    del sam
    torch.cuda.empty_cache()

    # Salvar estatísticas
    stats_path = output_dir / "inference_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    slope_removed = stats.get("slope_removed", 0)
    print(f"  Processados: {stats['processed']} | "
          f"Com feições: {stats['with_features']} | "
          f"Removidos por sombra: {stats['shadow_removed']} | "
          f"Removidos por slope: {slope_removed} | "
          f"Skipped: {stats['skipped']}")

    return stats


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Inferência em larga escala com SAM fine-tuned"
    )
    parser.add_argument(
        "--feature", type=str, default=None,
        choices=list(Config.FEATURES.keys()),
        help="Feição específica. Se omitido, processa todas."
    )
    parser.add_argument(
        "--year", type=int, default=None,
        choices=Config.YEARS,
        help="Ano específico. Se omitido, processa todos."
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3,
        help="Threshold de confiança para binarização (default: 0.3)"
    )
    parser.add_argument(
        "--no-shadow", action="store_true",
        help="Desativar subtração de sombra DEM (para comparação A/B)"
    )
    parser.add_argument(
        "--shadow-threshold", type=int, default=None,
        help="Threshold de hillshade para sombra (default: Config.SHADOW_HILLSHADE_THRESHOLD)"
    )
    parser.add_argument(
        "--annotated-only", action="store_true",
        help="Processar apenas tiles com ground truth (modo rápido para tuning/validação)"
    )
    parser.add_argument(
        "--pred-iou-threshold", type=float, default=0.5,
        help="Limiar mínimo de IoU predito por prompt SAM (default: 0.5)"
    )
    parser.add_argument(
        "--combine-mode", type=str, default="vote",
        choices=["max", "mean", "vote"],
        help="Combinação dos prompts: vote (recomendado), mean (conservador), max (sensível)"
    )
    parser.add_argument(
        "--vote-threshold", type=int, default=3,
        help="Mínimo de pontos concordantes para ativar pixel no modo vote (default: 3)"
    )
    parser.add_argument(
        "--no-feature-filter", action="store_true",
        help="Desativa pós-filtro por feição (debug rápido de overfitting/filtro)"
    )
    parser.add_argument(
        "--lakes-blue-ratio", type=float, default=1.2,
        help="Filtro lagos: razão mínima B/R (default: 1.2)"
    )
    parser.add_argument(
        "--lakes-dark-brightness", type=float, default=80.0,
        help="Filtro lagos: brilho máximo para água escura (default: 80)"
    )
    parser.add_argument(
        "--lakes-max-brightness", type=float, default=200.0,
        help="Filtro lagos: brilho máximo permitido (default: 200)"
    )
    parser.add_argument(
        "--lakes-preset", type=str, default="default",
        choices=["default", "precision"],
        help="Preset para lagos: default ou precision (mais conservador)"
    )
    parser.add_argument(
        "--crevasses-max-brightness", type=float, default=150.0,
        help="Filtro crevasses: brilho máximo (default: 150)"
    )
    parser.add_argument(
        "--crevasses-min-aspect", type=float, default=3.0,
        help="Filtro crevasses: aspect ratio mínimo (default: 3.0)"
    )
    parser.add_argument(
        "--channels-min-aspect", type=float, default=5.0,
        help="Filtro channels: aspect ratio mínimo (default: 5.0)"
    )
    parser.add_argument(
        "--tta", action="store_true", default=Config.USE_TTA,
        help="Ativar Test-Time Augmentation (4x mais lento, +2-4%% IoU)"
    )
    parser.add_argument(
        "--no-tta", action="store_true",
        help="Desativar TTA mesmo se ativo no config"
    )
    parser.add_argument(
        "--refinement", action="store_true", default=Config.USE_MASK_REFINEMENT,
        help="Ativar refinamento 2-pass com mask prompt"
    )
    parser.add_argument(
        "--no-refinement", action="store_true",
        help="Desativar refinamento 2-pass"
    )
    parser.add_argument(
        "--slope-filter", action="store_true", default=Config.USE_SLOPE_FILTER,
        help="Ativar filtro de slope (DEM) para lakes"
    )
    parser.add_argument(
        "--no-slope-filter", action="store_true",
        help="Desativar filtro de slope"
    )
    parser.add_argument(
        "--propose-refine", action="store_true",
        help="Modo 'propor e refinar' para lakes: detector espectral encontra candidatos, "
             "SAM refina com bbox prompt (igual ao treino). Muito mais preciso que grade cega."
    )
    args = parser.parse_args()

    use_shadow = not args.no_shadow
    use_feature_filter = not args.no_feature_filter
    use_tta = args.tta and not args.no_tta
    use_refinement = args.refinement and not args.no_refinement
    use_slope_filter = args.slope_filter and not args.no_slope_filter

    # Preset focado em reduzir falsos positivos em lagos
    if args.lakes_preset == "precision":
        args.combine_mode = "vote"
        args.pred_iou_threshold = 0.85
        args.threshold = 0.55
        args.lakes_blue_ratio = 1.35
        args.lakes_dark_brightness = 70.0
        args.lakes_max_brightness = 180.0

    print("=" * 60)
    print("FASE 4 - INFERÊNCIA EM LARGA ESCALA")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Backend: {SAM_BACKEND}")
    print(f"Threshold: {args.threshold}")
    print(f"Pred IoU threshold: {args.pred_iou_threshold}")
    print(f"Combine mode: {args.combine_mode}")
    print(f"Feature filter: {'SIM' if use_feature_filter else 'NÃO'}")
    print(f"Lakes preset: {args.lakes_preset}")
    print(f"Lakes spectral: blue/red>{args.lakes_blue_ratio:.2f}, "
          f"dark brightness<{args.lakes_dark_brightness:.1f}, "
          f"max brightness<{args.lakes_max_brightness:.1f}")
    print(f"Crevasses filter: brightness<{args.crevasses_max_brightness:.1f}, "
          f"aspect>{args.crevasses_min_aspect:.2f}")
    print(f"Channels filter: aspect>{args.channels_min_aspect:.2f}")
    print(f"Subtração de sombra: {'SIM' if use_shadow else 'NÃO'}")
    print(f"TTA: {'SIM (' + str(Config.TTA_TRANSFORMS) + ')' if use_tta else 'NÃO'}")
    print(f"Refinamento 2-pass: {'SIM' if use_refinement else 'NÃO'}")
    print(f"Slope filter (lakes): {'SIM (max ' + str(Config.SLOPE_FILTER_MAX_DEGREES) + ' graus)' if use_slope_filter else 'NÃO'}")
    print(f"Propose-refine (lakes): {'SIM' if args.propose_refine else 'NÃO'}")
    print(f"Modo rápido (annotated-only): {'SIM' if args.annotated_only else 'NÃO'}")
    print(f"Otimização: encoder float16 + limpeza VRAM por tile")

    start = time.time()

    features = [args.feature] if args.feature else list(Config.FEATURES.keys())
    years = [args.year] if args.year else Config.YEARS

    all_stats = {}
    for feature in features:
        for year in years:
            key = f"{feature}_{year}"
            all_stats[key] = run_inference(
                feature, year, args.threshold,
                use_shadow=use_shadow,
                shadow_threshold=args.shadow_threshold,
                annotated_only=args.annotated_only,
                pred_iou_threshold=args.pred_iou_threshold,
                combine_mode=args.combine_mode,
                vote_threshold=args.vote_threshold,
                use_feature_filter=use_feature_filter,
                lake_blue_ratio=args.lakes_blue_ratio,
                lake_dark_brightness=args.lakes_dark_brightness,
                lake_max_brightness=args.lakes_max_brightness,
                crevasses_max_brightness=args.crevasses_max_brightness,
                crevasses_min_aspect=args.crevasses_min_aspect,
                channels_min_aspect=args.channels_min_aspect,
                use_tta=use_tta,
                use_refinement=use_refinement,
                use_slope_filter=use_slope_filter,
                propose_refine=args.propose_refine,
            )

    # Resumo final
    print(f"\n{'='*60}")
    print("RESUMO DA INFERÊNCIA")
    print(f"{'='*60}")
    for key, stats in all_stats.items():
        if stats:
            pct = 100.0 * stats["with_features"] / max(stats["processed"], 1)
            print(f"  {key}: {stats['with_features']}/{stats['processed']} "
                  f"tiles com feições ({pct:.1f}%)")

    elapsed = time.time() - start
    print(f"\nTempo total: {elapsed/60:.1f} minutos")


if __name__ == "__main__":
    main()
