"""
shadow_utils.py - Detecção de sombra topográfica usando DEM

Projeto: LACRIO IC - Extração de Feições Supraglaciais
Objetivo: Gerar máscaras de sombra a partir do DEM para remover falsos positivos
          na detecção de lagos supraglaciais (sombras confundidas com água).

Método:
  1. Hillshade (Horn, 1981) - mesmo algoritmo do GDAL gdaldem
  2. Múltiplos ângulos solares típicos para ~54°S (verão austral)
  3. Interseção conservadora: pixel = sombra somente se em sombra em TODOS os ângulos
  4. Filtro de textura: sombras são uniformes (baixa variância), lagos têm reflexos

Uso:
    from shadow_utils import precompute_year_shadows, get_shadow_mask_for_tile

    shadow_mask, dem_transform = precompute_year_shadows(2016)
    tile_shadow = get_shadow_mask_for_tile(tile_info, shadow_mask, dem_transform)
"""

import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds
from pathlib import Path

from config import Config


def compute_hillshade(dem, res_x, res_y, azimuth=0.0, altitude=40.0):
    """Computa hillshade a partir de um DEM usando o método de Horn (1981).

    Mesmo algoritmo usado pelo GDAL (gdaldem hillshade).

    Args:
        dem: Array 2D de elevação (metros). NaN = NoData.
        res_x: Tamanho do pixel em X (metros).
        res_y: Tamanho do pixel em Y (metros), valor positivo.
        azimuth: Azimute solar em graus (0=Norte, sentido horário).
        altitude: Elevação solar em graus acima do horizonte.

    Returns:
        hillshade: Array 2D uint8 [0, 255]. 0=sombra total, 255=iluminação máxima.
    """
    # Gradientes de elevação
    dy, dx = np.gradient(dem, res_y, res_x)

    # Slope e aspect
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)

    # Converter ângulos solares para radianos
    zenith = np.radians(90.0 - altitude)
    azimuth_rad = np.radians(azimuth)

    # Hillshade (Horn's formula)
    hs = (np.cos(zenith) * np.cos(slope) +
          np.sin(zenith) * np.sin(slope) * np.cos(azimuth_rad - aspect))

    # Normalizar para [0, 255]
    hs = np.clip(hs * 255.0, 0, 255).astype(np.uint8)

    return hs


def generate_shadow_mask(dem, res_x, res_y, threshold=None,
                         azimuths=None, altitudes=None):
    """Gera máscara binária de sombra a partir do DEM.

    Computa hillshade em múltiplas combinações de ângulos solares e aplica
    interseção conservadora: pixel é sombra somente se estiver em sombra
    em TODAS as combinações.

    Args:
        dem: Array 2D de elevação (metros).
        res_x: Tamanho do pixel em X (metros).
        res_y: Tamanho do pixel em Y (metros), positivo.
        threshold: Valor de hillshade abaixo do qual = sombra (0-255).
        azimuths: Lista de azimutes solares em graus.
        altitudes: Lista de altitudes solares em graus.

    Returns:
        shadow_mask: Array 2D uint8 (255=sombra, 0=não sombra).
    """
    if threshold is None:
        threshold = Config.SHADOW_HILLSHADE_THRESHOLD
    if azimuths is None:
        azimuths = Config.SHADOW_SOLAR_AZIMUTHS
    if altitudes is None:
        altitudes = Config.SHADOW_SOLAR_ALTITUDES

    # Tratar NoData
    nodata_mask = np.isnan(dem)
    dem_clean = np.nan_to_num(dem, nan=0.0)

    # Interseção: começa com tudo em sombra, remove se iluminado em qualquer ângulo
    shadow = np.ones(dem.shape, dtype=bool)

    for az in azimuths:
        for alt in altitudes:
            hs = compute_hillshade(dem_clean, res_x, res_y, az, alt)
            # Se iluminado neste ângulo, não é sombra
            shadow &= (hs < threshold)

    # NoData não é sombra
    shadow[nodata_mask] = False

    return (shadow.astype(np.uint8) * 255)


def precompute_year_shadows(year, threshold=None):
    """Pré-computa máscara de sombra completa para um ano.

    Abre o DEM uma única vez, computa a máscara de sombra em resolução
    do DEM, e retorna para uso repetido durante a inferência.

    Args:
        year: Ano do DEM a usar.
        threshold: Threshold de hillshade (usa Config se None).

    Returns:
        shadow_mask: Array 2D uint8 (255=sombra, 0=não sombra) em resolução DEM.
        dem_transform: Affine transform do DEM (rasterio).
        None, None se o DEM não existir.
    """
    try:
        dem_path = Config.get_dem_path(year)
    except ValueError:
        print(f"  [SOMBRA] DEM não disponível para {year}")
        return None, None

    if not dem_path.exists():
        print(f"  [SOMBRA] Arquivo DEM não encontrado: {dem_path}")
        return None, None

    print(f"  [SOMBRA] Carregando DEM {year}: {dem_path.name}")

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        dem_transform = src.transform
        dem_crs = src.crs
        res_x = abs(src.res[0])
        res_y = abs(src.res[1])

        # Marcar NoData
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan

    print(f"  [SOMBRA] DEM shape: {dem.shape}, resolução: {res_x:.3f}m x {res_y:.3f}m")
    print(f"  [SOMBRA] Computando hillshade em {len(Config.SHADOW_SOLAR_AZIMUTHS)}x"
          f"{len(Config.SHADOW_SOLAR_ALTITUDES)} ângulos...")

    shadow_mask = generate_shadow_mask(dem, res_x, res_y, threshold)

    coverage = (shadow_mask > 127).sum() / max(shadow_mask.size, 1)
    print(f"  [SOMBRA] Cobertura de sombra: {coverage:.1%}")

    return shadow_mask, dem_transform


def get_shadow_mask_for_tile(tile_info, shadow_mask, dem_transform,
                             tile_size=512):
    """Extrai máscara de sombra para um tile específico.

    Mapeia a extensão geográfica do tile para a máscara de sombra
    pré-computada (em resolução DEM) e redimensiona para o tamanho do tile.

    Args:
        tile_info: Dict do tiles_index.json com chave 'transform'
                   (6 elementos: [a, b, c, d, e, f]).
        shadow_mask: Máscara de sombra completa do DEM (de precompute_year_shadows).
        dem_transform: Affine transform do DEM.
        tile_size: Dimensão do tile em pixels (512).

    Returns:
        tile_shadow: Array (tile_size, tile_size) uint8 (255=sombra, 0=não sombra).
    """
    if shadow_mask is None:
        return np.zeros((tile_size, tile_size), dtype=np.uint8)

    # Bounding box do tile em coordenadas UTM
    t = tile_info["transform"]
    # t = [pixel_size_x, rotation_b, origin_x, rotation_d, pixel_size_y, origin_y]
    xmin = t[2]
    xmax = t[2] + tile_size * t[0]
    ymax = t[5]
    ymin = t[5] + tile_size * t[4]  # t[4] é negativo

    # Converter bounds para janela no DEM
    try:
        window = from_bounds(xmin, ymin, xmax, ymax, dem_transform)
    except Exception:
        return np.zeros((tile_size, tile_size), dtype=np.uint8)

    # Extrair indices inteiros da janela
    row_start = max(0, int(window.row_off))
    row_stop = min(shadow_mask.shape[0], int(window.row_off + window.height))
    col_start = max(0, int(window.col_off))
    col_stop = min(shadow_mask.shape[1], int(window.col_off + window.width))

    # Verificar se a janela é válida
    if row_stop <= row_start or col_stop <= col_start:
        return np.zeros((tile_size, tile_size), dtype=np.uint8)

    # Extrair patch da máscara de sombra
    patch = shadow_mask[row_start:row_stop, col_start:col_stop]

    # Redimensionar para tamanho do tile (DEM ~22cm → tile ~7.8cm)
    tile_shadow = cv2.resize(patch, (tile_size, tile_size),
                             interpolation=cv2.INTER_NEAREST)

    return tile_shadow


def compute_texture_variance(image, kernel_size=15):
    """Computa variância local de textura da imagem.

    Sombras têm textura uniforme (baixa variância).
    Lagos podem ter reflexos e ondulações (variância mais alta).

    Args:
        image: Imagem RGB (H, W, 3) uint8.
        kernel_size: Tamanho da janela para computar variância local.

    Returns:
        variance_map: Array (H, W) float32 com variância local.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Variância = E[X²] - E[X]²
    mean = cv2.blur(gray, (kernel_size, kernel_size))
    mean_sq = cv2.blur(gray**2, (kernel_size, kernel_size))
    variance = mean_sq - mean**2

    # Garantir não-negativo (erros numéricos)
    variance = np.maximum(variance, 0.0)

    return variance


def filter_by_texture(mask, image, min_variance=None):
    """Remove componentes conectados com textura muito uniforme (provável sombra).

    Args:
        mask: Máscara binária (H, W) uint8.
        image: Imagem RGB (H, W, 3) uint8.
        min_variance: Variância mínima para manter componente.

    Returns:
        filtered: Máscara filtrada (H, W) uint8.
    """
    if min_variance is None:
        min_variance = Config.SHADOW_TEXTURE_MIN_VARIANCE

    if mask.max() == 0:
        return mask

    variance_map = compute_texture_variance(image)

    filtered = np.zeros_like(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    for label_id in range(1, num_labels):
        component_mask = (labels == label_id)
        mean_var = variance_map[component_mask].mean()
        if mean_var >= min_variance:
            filtered[component_mask] = 255

    return filtered


# ============================================================================
# Filtro por relief topografico (discrimina fenda vs detrito escuro)
# ============================================================================

def compute_relief_map(dem, window_pixels):
    """Computa relief local: z - mean(z em janela).

    Fendas sao depressoes verticais lineares: relief fortemente negativo.
    Detritos escuros sao camadas superficiais: relief proximo de zero.

    Args:
        dem: Array 2D de elevacao (metros). NaN = NoData.
        window_pixels: Tamanho impar da janela (em pixels do DEM) para a media local.

    Returns:
        relief: Array 2D float32 (metros). Negativo = depressao local.
    """
    nodata_mask = np.isnan(dem)
    dem_clean = np.nan_to_num(dem, nan=0.0)

    mean = cv2.blur(dem_clean, (window_pixels, window_pixels))
    relief = dem_clean - mean
    relief[nodata_mask] = 0.0
    return relief.astype(np.float32)


def precompute_year_relief(year, window_meters=2.5):
    """Pre-computa relief map para um ano.

    Args:
        year: Ano do DEM.
        window_meters: Tamanho da janela em metros para media local
                       (~2-3m e razoavel: contexto suficiente sem diluir).

    Returns:
        relief_map: Array 2D float32 (metros) em resolucao do DEM.
        dem_transform: Affine transform do DEM (rasterio).
        None, None se DEM nao existir.
    """
    try:
        dem_path = Config.get_dem_path(year)
    except (ValueError, AttributeError):
        print(f"  [DEM] DEM nao disponivel para {year}")
        return None, None

    if not dem_path.exists():
        print(f"  [DEM] Arquivo DEM nao encontrado: {dem_path}")
        return None, None

    print(f"  [DEM] Carregando DEM {year}: {dem_path.name}")

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        dem_transform = src.transform
        res_x = abs(src.res[0])
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan

    window_pixels = max(3, int(round(window_meters / res_x)))
    if window_pixels % 2 == 0:
        window_pixels += 1

    print(f"  [DEM] Res: {res_x:.3f}m | janela: {window_pixels}px (~{window_pixels*res_x:.1f}m)")

    relief = compute_relief_map(dem, window_pixels)

    valid = relief[~np.isnan(relief) & (relief != 0)]
    if len(valid) > 0:
        print(f"  [DEM] Relief: min={valid.min():.2f}m  "
              f"p5={np.percentile(valid, 5):.2f}m  "
              f"median={np.median(valid):.2f}m  "
              f"max={valid.max():.2f}m")

    return relief, dem_transform


def get_relief_for_tile(tile_info, relief_map, dem_transform, tile_size=512):
    """Extrai relief de um tile especifico.

    Mesma logica de get_shadow_mask_for_tile mas para float32 (relief em metros).
    """
    if relief_map is None:
        return np.zeros((tile_size, tile_size), dtype=np.float32)

    t = tile_info["transform"]
    xmin = t[2]
    xmax = t[2] + tile_size * t[0]
    ymax = t[5]
    ymin = t[5] + tile_size * t[4]

    try:
        window = from_bounds(xmin, ymin, xmax, ymax, dem_transform)
    except Exception:
        return np.zeros((tile_size, tile_size), dtype=np.float32)

    row_start = max(0, int(window.row_off))
    row_stop = min(relief_map.shape[0], int(window.row_off + window.height))
    col_start = max(0, int(window.col_off))
    col_stop = min(relief_map.shape[1], int(window.col_off + window.width))

    if row_stop <= row_start or col_stop <= col_start:
        return np.zeros((tile_size, tile_size), dtype=np.float32)

    patch = relief_map[row_start:row_stop, col_start:col_stop]
    tile_relief = cv2.resize(patch, (tile_size, tile_size),
                             interpolation=cv2.INTER_LINEAR)
    return tile_relief


def filter_by_dem_relief(mask, tile_relief, min_depth=-0.3, min_frac_below=0.30):
    """Remove componentes sem assinatura topografica de fenda.

    Componente e mantido se:
      - relief MEDIO dentro do componente <= min_depth (depressao consistente)
      OU
      - pelo menos `min_frac_below` da area tem relief <= min_depth (fenda parcialmente
        capturada pela predicao, mas com nucleo afundado)

    Args:
        mask: Mascara binaria predita uint8 (H, W).
        tile_relief: Relief map do tile float32 (m, H, W).
        min_depth: Profundidade minima (m, negativo). Default -0.3m.
                   Fendas tipicamente > 50cm de profundidade local.
        min_frac_below: Fracao minima de pixels do componente abaixo do threshold.

    Returns:
        filtered: Mascara filtrada uint8.
    """
    if mask.max() == 0:
        return mask

    filtered = np.zeros_like(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for label_id in range(1, num_labels):
        component_mask = (labels == label_id)
        relief_vals = tile_relief[component_mask]

        if len(relief_vals) == 0:
            continue

        mean_relief = float(relief_vals.mean())
        frac_below = float((relief_vals <= min_depth).mean())

        if mean_relief <= min_depth or frac_below >= min_frac_below:
            filtered[component_mask] = 255

    return filtered


# ============================================================================
# DEM como canais de input para a rede (relief + slope + curvature)
# ============================================================================

def compute_slope_map(dem, res_x, res_y):
    """Slope em graus a partir do DEM.

    Args:
        dem: Array 2D de elevacao (m), NaN = NoData.
        res_x, res_y: Resolucoes em metros (res_y positivo).

    Returns:
        slope: Array 2D float32 (graus). 0 = plano, 90 = vertical.
    """
    nodata_mask = np.isnan(dem)
    dem_clean = np.nan_to_num(dem, nan=0.0)
    dy, dx = np.gradient(dem_clean, res_y, res_x)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)
    slope_deg[nodata_mask] = 0.0
    return slope_deg


def compute_curvature_map(dem, res_x, res_y, smooth_sigma=1.0):
    """Curvatura (Laplaciano) do DEM.

    Fendas tem curvatura negativa concentrada (depressao).
    Cristas tem curvatura positiva. Plano tem ~0.

    Args:
        dem: Array 2D de elevacao (m).
        res_x, res_y: Resolucoes em metros.
        smooth_sigma: Sigma do filtro Gaussiano de pre-suavizacao
                      (reduz ruido de alta frequencia do DEM).

    Returns:
        curvature: Array 2D float32 (m^-1).
    """
    nodata_mask = np.isnan(dem)
    dem_clean = np.nan_to_num(dem, nan=0.0).astype(np.float32)

    if smooth_sigma > 0:
        ksize = max(3, int(round(smooth_sigma * 6)) | 1)  # impar
        dem_clean = cv2.GaussianBlur(dem_clean, (ksize, ksize), smooth_sigma)

    # Laplaciano (2a derivada) em coordenadas metricas
    dy = np.gradient(dem_clean, res_y, axis=0)
    dyy = np.gradient(dy, res_y, axis=0)
    dx = np.gradient(dem_clean, res_x, axis=1)
    dxx = np.gradient(dx, res_x, axis=1)
    curvature = (dxx + dyy).astype(np.float32)
    curvature[nodata_mask] = 0.0
    return curvature


def precompute_year_dem_features(year, window_meters=3.0,
                                  features=("relief", "slope", "curvature")):
    """Pre-computa stack de features DEM para um ano.

    Returns:
        features_stack: Array 3D float32 (H, W, n_features) ou None.
        dem_transform: Affine transform do DEM.
        feature_names: Lista de nomes das features na ordem dos canais.
        stats: Dict {feature: (mean, std)} para normalizacao posterior.
    """
    try:
        dem_path = Config.get_dem_path(year)
    except (ValueError, AttributeError):
        print(f"  [DEM-CH] DEM nao disponivel para {year}")
        return None, None, None, None

    if not dem_path.exists():
        print(f"  [DEM-CH] Arquivo DEM nao encontrado: {dem_path}")
        return None, None, None, None

    print(f"  [DEM-CH] Carregando DEM {year}: {dem_path.name}")

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        dem_transform = src.transform
        res_x = abs(src.res[0])
        res_y = abs(src.res[1])
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan

    window_pixels = max(3, int(round(window_meters / res_x)))
    if window_pixels % 2 == 0:
        window_pixels += 1

    print(f"  [DEM-CH] Res: {res_x:.3f}m | janela relief: {window_pixels}px"
          f" (~{window_pixels*res_x:.1f}m) | features: {features}")

    maps = {}
    if "relief" in features:
        maps["relief"] = compute_relief_map(dem, window_pixels)
    if "slope" in features:
        maps["slope"] = compute_slope_map(dem, res_x, res_y)
    if "curvature" in features:
        maps["curvature"] = compute_curvature_map(dem, res_x, res_y)

    # Stats (mean/std) por feature, para z-score
    # Mascarar valores extremos de bordas do glaciar (NoData propagado)
    stats = {}
    for name, arr in maps.items():
        # Robusto: usar percentis 1-99 para evitar bordas extremas
        valid = arr[(arr != 0) | (np.abs(arr) < 1e-6)]  # inclui zeros (legitimos)
        if len(valid) == 0:
            valid = arr.flatten()
        p1, p99 = np.percentile(valid, [1, 99])
        clipped = np.clip(arr, p1, p99)
        m = float(clipped.mean())
        s = float(clipped.std()) + 1e-6
        stats[name] = (m, s)
        print(f"  [DEM-CH] {name}: mean={m:.4f} std={s:.4f} (clip {p1:.3f}..{p99:.3f})")

    # Empilhar como (H, W, N)
    stack = np.stack([maps[f] for f in features if f in maps], axis=-1).astype(np.float32)

    return stack, dem_transform, list(features), stats


def get_dem_features_for_tile(tile_info, features_stack, dem_transform,
                              tile_size=512):
    """Extrai patch (tile_size, tile_size, n_features) de DEM features para um tile."""
    if features_stack is None:
        return None

    n_feat = features_stack.shape[2]
    t = tile_info["transform"]
    xmin = t[2]
    xmax = t[2] + tile_size * t[0]
    ymax = t[5]
    ymin = t[5] + tile_size * t[4]

    try:
        window = from_bounds(xmin, ymin, xmax, ymax, dem_transform)
    except Exception:
        return np.zeros((tile_size, tile_size, n_feat), dtype=np.float32)

    row_start = max(0, int(window.row_off))
    row_stop = min(features_stack.shape[0], int(window.row_off + window.height))
    col_start = max(0, int(window.col_off))
    col_stop = min(features_stack.shape[1], int(window.col_off + window.width))

    if row_stop <= row_start or col_stop <= col_start:
        return np.zeros((tile_size, tile_size, n_feat), dtype=np.float32)

    patch = features_stack[row_start:row_stop, col_start:col_stop, :]
    resized = cv2.resize(patch, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = resized[..., None]
    return resized.astype(np.float32)


def normalize_dem_features(tile_features, stats, feature_names, clip_sigma=6.0):
    """Z-score por canal usando stats pre-computados, com clip de seguranca.

    O clip impede que bordas mal tratadas (descontinuidades NoData->0 em
    curvature/slope) injetem valores absurdos no input da rede.

    Args:
        tile_features: Array (H, W, n_feat) float32.
        stats: Dict {feature_name: (mean, std)}.
        feature_names: Lista de nomes (mesma ordem dos canais).
        clip_sigma: Clip simetrico em multiplos de sigma. None = sem clip.
    """
    out = tile_features.copy()
    for i, name in enumerate(feature_names):
        if name in stats:
            m, s = stats[name]
            out[..., i] = (out[..., i] - m) / s
            if clip_sigma is not None:
                out[..., i] = np.clip(out[..., i], -clip_sigma, clip_sigma)
    return out


def build_dem_provider(years_list, feature_names=("relief", "slope", "curvature"),
                       window_meters=3.0):
    """Pre-computa DEM features por ano e devolve callable que mapeia tile_path -> features.

    Returns:
        provider: Callable(tile_path: Path|str) -> ndarray (H, W, n_feat) ja
                  normalizado via z-score (clipado a +-6 sigma), ou None.
        feature_names: Lista efetiva de nomes (a entrada filtrada).
    """
    import json

    year_data = {}
    tile_index_by_path = {}

    for year in years_list:
        stack, transform, names, stats = precompute_year_dem_features(
            year, window_meters=window_meters, features=feature_names,
        )
        if stack is None:
            print(f"  [DEM-CH] AVISO: pulando ano {year} (DEM nao disponivel)")
            continue
        year_data[year] = (stack, transform, names, stats)

        tiles_index_path = Config.TILES_DIR / str(year) / "tiles_index.json"
        if tiles_index_path.exists():
            with open(tiles_index_path) as fh:
                idx = json.load(fh)
            year_dir = Config.TILES_DIR / str(year)
            for t in idx["tiles"]:
                tile_index_by_path[str(year_dir / t["filename"])] = (year, t)

    if not year_data:
        return None, None

    def provider(tile_path):
        info = tile_index_by_path.get(str(tile_path))
        if info is None:
            return None
        year, tile_info = info
        stack, transform, names, stats = year_data[year]
        feats = get_dem_features_for_tile(tile_info, stack, transform)
        return normalize_dem_features(feats, stats, names)

    return provider, list(feature_names)


def get_shadow_coverage_for_tiles(year):
    """Calcula cobertura de sombra para todos os tiles de um ano.

    Usado para selecionar hard negatives de sombra para treinamento.

    Args:
        year: Ano para processar.

    Returns:
        Dict mapeando tile_id (str) → cobertura de sombra (float 0-1).
        Dict vazio se DEM não disponível.
    """
    import json

    shadow_mask, dem_transform = precompute_year_shadows(year)
    if shadow_mask is None:
        return {}

    tiles_index_path = Config.TILES_DIR / str(year) / "tiles_index.json"
    if not tiles_index_path.exists():
        return {}

    with open(tiles_index_path) as f:
        tiles_index = json.load(f)

    coverage = {}
    for tile_info in tiles_index["tiles"]:
        tile_id = f"tile_{tile_info['id']:06d}"
        tile_shadow = get_shadow_mask_for_tile(
            tile_info, shadow_mask, dem_transform
        )
        cov = (tile_shadow > 127).sum() / max(tile_shadow.size, 1)
        if cov > 0.01:  # Ignorar tiles com sombra negligível
            coverage[tile_id] = float(cov)

    return coverage


# ============================================================================
# Teste standalone
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DE DETECÇÃO DE SOMBRA")
    print("=" * 60)

    shadow_mask, dem_transform = precompute_year_shadows(2016)
    if shadow_mask is not None:
        Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Salvar como GeoTIFF (mesmo CRS e transform do DEM)
        output_path = Config.RESULTS_DIR / "shadow_mask_2016.tif"
        dem_path = Config.get_dem_path(2016)
        with rasterio.open(dem_path) as src:
            profile = src.profile.copy()
            profile.update(count=1, dtype="uint8", compress="lzw", nodata=0)

        with rasterio.open(str(output_path), "w", **profile) as dst:
            dst.write(shadow_mask, 1)

        print(f"\n  Máscara de sombra salva: {output_path}")
        print(f"  Shape: {shadow_mask.shape}")
        print(f"  CRS: {profile['crs']}")
        print(f"  Abra em QGIS sobreposta ao mosaico para verificar alinhamento.")

        # Testar extração por tile
        import json
        tiles_index_path = Config.TILES_DIR / "2016" / "tiles_index.json"
        if tiles_index_path.exists():
            with open(tiles_index_path) as f:
                tiles_index = json.load(f)

            tile_info = tiles_index["tiles"][0]
            tile_shadow = get_shadow_mask_for_tile(
                tile_info, shadow_mask, dem_transform
            )
            print(f"\n  Tile 0 shadow shape: {tile_shadow.shape}")
            cov = (tile_shadow > 127).sum() / tile_shadow.size
            print(f"  Tile 0 shadow coverage: {cov:.1%}")
