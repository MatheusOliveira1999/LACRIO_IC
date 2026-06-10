"""
05_reconstruct_mosaic.py - Reconstrução de mosaicos GeoTIFF a partir dos tiles

Projeto: LACRIO IC - Extração de Feições Supraglaciais
Fase 5: Combinar máscaras individuais de volta em mosaicos georreferenciados

Uso:
    python 05_reconstruct_mosaic.py                              # Todos
    python 05_reconstruct_mosaic.py --feature lakes --year 2016  # Específico
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from tqdm import tqdm

from config import Config


def reconstruct_mosaic(feature: str, year: int, masks_subdir_suffix: str = "",
                       output_suffix: str = ""):
    """Reconstrói mosaico GeoTIFF a partir das máscaras individuais dos tiles.

    Algoritmo:
    1. Carrega tiles_index.json para obter posições e metadados
    2. Cria array vazio com dimensões do mosaico original
    3. Para cada tile com máscara predita, coloca no array na posição correta
    4. Em regiões de sobreposição, usa operador máximo (merge conservador)
    5. Salva como GeoTIFF com CRS e geotransform originais

    Args:
        feature: Nome da feição (lakes, crevasses, channels).
        year: Ano do mosaico.
    """
    print(f"\n--- Reconstrução: {feature} | Ano: {year} ---")

    # Carregar índice de tiles
    tiles_index_path = Config.TILES_DIR / str(year) / "tiles_index.json"
    if not tiles_index_path.exists():
        print(f"  [ERRO] Índice de tiles não encontrado: {tiles_index_path}")
        return

    with open(tiles_index_path) as f:
        tiles_index = json.load(f)

    # Obter dimensões do mosaico original
    original_width = tiles_index["original_width"]
    original_height = tiles_index["original_height"]
    crs = tiles_index["crs"]

    print(f"  Mosaico original: {original_width} x {original_height} pixels")
    print(f"  CRS: {crs}")

    # Diretório com máscaras preditas
    masks_dir = Config.MASKS_DIR / str(year) / f"{feature}{masks_subdir_suffix}"
    if not masks_dir.exists():
        print(f"  [ERRO] Diretório de máscaras não encontrado: {masks_dir}")
        return

    # Listar máscaras disponíveis
    mask_files = list(masks_dir.glob(f"tile_*_{feature}.png"))
    if not mask_files:
        print(f"  [AVISO] Nenhuma máscara encontrada em {masks_dir}")
        return

    print(f"  Máscaras encontradas: {len(mask_files)}")

    # Criar mapa de ID -> posição no mosaico
    tile_positions = {}
    for tile_info in tiles_index["tiles"]:
        tile_positions[tile_info["id"]] = tile_info

    # Criar array do mosaico (uint8, processado em blocos para economizar RAM)
    mosaic = np.zeros((original_height, original_width), dtype=np.uint8)

    placed = 0
    for mask_file in tqdm(mask_files, desc=f"  Montando {feature}/{year}"):
        # Extrair tile ID do nome do arquivo
        stem = mask_file.stem  # ex: tile_000500_lakes
        tile_id_str = stem.split("_")[1]  # ex: 000500
        tile_id = int(tile_id_str)

        if tile_id not in tile_positions:
            continue

        pos = tile_positions[tile_id]
        x = pos["x"]
        y = pos["y"]
        w = pos["width"]
        h = pos["height"]

        # Carregar máscara
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        # Garantir dimensões corretas
        mask = mask[:h, :w]

        # Calcular limites dentro do mosaico
        y_end = min(y + h, original_height)
        x_end = min(x + w, original_width)
        mask_h = y_end - y
        mask_w = x_end - x

        # Operador máximo para regiões de sobreposição
        mosaic[y:y_end, x:x_end] = np.maximum(
            mosaic[y:y_end, x:x_end],
            mask[:mask_h, :mask_w]
        )

        placed += 1

    print(f"  Tiles posicionados: {placed}/{len(mask_files)}")

    # Obter geotransform do mosaico original de referência
    mosaic_path = Config.get_mosaic_path(year)

    with rasterio.open(mosaic_path) as src:
        transform = src.transform
        src_crs = src.crs

    # Criar diretório de saída
    output_dir = Config.RESULTS_DIR / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Salvar GeoTIFF
    output_path = output_dir / f"{feature}_mask_{year}{output_suffix}.tif"

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": original_width,
        "height": original_height,
        "count": 1,
        "crs": src_crs,
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic, 1)
        dst.update_tags(
            feature=feature,
            year=str(year),
            source="SAM fine-tuned (LACRIO IC)",
            tiles_used=str(placed),
        )

    # Estatísticas
    feature_pixels = int((mosaic > 0).sum())
    total_pixels = int(mosaic.size)
    coverage_pct = 100.0 * feature_pixels / total_pixels

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"  GeoTIFF salvo: {output_path}")
    print(f"  Tamanho: {file_size_mb:.1f} MB")
    print(f"  Pixels com feição: {feature_pixels:,} ({coverage_pct:.2f}%)")

    # Salvar metadados
    meta = {
        "feature": feature,
        "year": year,
        "output_file": str(output_path),
        "crs": str(src_crs),
        "dimensions": [original_width, original_height],
        "tiles_used": placed,
        "feature_pixels": feature_pixels,
        "total_pixels": total_pixels,
        "coverage_percent": round(coverage_pct, 4),
        "file_size_mb": round(file_size_mb, 2),
    }

    meta_path = output_dir / f"{feature}_mask_{year}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reconstrução de mosaicos GeoTIFF a partir dos tiles"
    )
    parser.add_argument(
        "--feature", type=str, default=None,
        choices=list(Config.FEATURES.keys()),
        help="Feição específica. Se omitido, reconstrói todas."
    )
    parser.add_argument(
        "--year", type=int, default=None,
        choices=Config.YEARS,
        help="Ano específico. Se omitido, reconstrói todos."
    )
    parser.add_argument(
        "--tiles-dir", type=Path, default=None,
        help="Diretório alternativo de tiles (default: Config.TILES_DIR)."
    )
    parser.add_argument(
        "--masks-subdir-suffix", type=str, default="",
        help="Sufixo da subpasta de máscaras (ex.: '_8cm' -> masks/{ano}/crevasses_8cm/)."
    )
    parser.add_argument(
        "--output-suffix", type=str, default="",
        help="Sufixo no nome do GeoTIFF de saída (ex.: '_8cm' -> crevasses_mask_2016_8cm.tif)."
    )
    args = parser.parse_args()

    if args.tiles_dir is not None:
        Config.TILES_DIR = args.tiles_dir.resolve()
        print(f"📁 Tiles vindos de: {Config.TILES_DIR}")

    print("=" * 60)
    print("FASE 5 - RECONSTRUÇÃO DE MOSAICOS GEOTIFF")
    print("=" * 60)

    start = time.time()

    features = [args.feature] if args.feature else list(Config.FEATURES.keys())
    years = [args.year] if args.year else Config.YEARS

    results = {}
    for feature in features:
        for year in years:
            meta = reconstruct_mosaic(
                feature, year,
                masks_subdir_suffix=args.masks_subdir_suffix,
                output_suffix=args.output_suffix,
            )
            if meta:
                results[f"{feature}_{year}"] = meta

    # Resumo
    print(f"\n{'='*60}")
    print("RESUMO DA RECONSTRUÇÃO")
    print(f"{'='*60}")
    for key, meta in results.items():
        print(f"  {key}: {meta['coverage_percent']:.2f}% cobertura | "
              f"{meta['file_size_mb']:.1f} MB")

    elapsed = time.time() - start
    print(f"\nTempo total: {elapsed/60:.1f} minutos")


if __name__ == "__main__":
    main()
