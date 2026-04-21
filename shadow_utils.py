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
