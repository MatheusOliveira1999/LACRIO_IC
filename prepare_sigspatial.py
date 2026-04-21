"""
prepare_sigspatial.py - Prepara o dataset SIGSPATIAL Cup 2023 para pre-treinamento SAM

Converte as imagens GeoTIFF grandes + lake_polygons_training.gpkg em pares
de tiles (512x512 RGB) e mascaras binarias, prontos para 03a_pretrain_satellite.py.

Uso:
    python prepare_sigspatial.py

Saida:
    data/SIGSPATIAL/tiles/   -> tiles RGB .png (512x512)
    data/SIGSPATIAL/masks/   -> mascaras binarias .png (512x512, 0/255)
"""

import argparse
import numpy as np
from pathlib import Path

import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import geopandas as gpd
from PIL import Image


# ============================================================================
# Configuracao
# ============================================================================

TILE_SIZE = 512
STRIDE = 384       # overlap de 128px para mais cobertura de lagos
MIN_VALID_RATIO = 0.8   # minimo de pixels nao-pretos
MIN_LAKE_TILES_RATIO = 0.3  # proporcao minima de tiles com lago (controle de balanco)


def find_tif_for_image(image_name: str, search_dir: Path) -> Path:
    """Encontra o arquivo TIF correspondente a um nome de imagem do GeoPackage.

    Os nomes no GeoPackage sao como 'Greenland26X_22W_Sentinel2_2019-07-31_25.tif'
    mas os arquivos baixados podem ter sufixo como '-004.tif'.
    """
    # Tentar match exato primeiro
    for tif in search_dir.rglob("*.tif"):
        if tif.name == image_name:
            return tif

    # Tentar match parcial (sem sufixo numerico)
    stem = Path(image_name).stem  # ex: Greenland26X_22W_Sentinel2_2019-07-31_25
    for tif in search_dir.rglob("*.tif"):
        if tif.stem.startswith(stem):
            return tif

    return None


def prepare_tiles(sigspatial_dir: Path, output_dir: Path = None,
                  max_negative_ratio: float = 1.0):
    """Converte GeoTIFFs + GeoPackage em tiles/masks para pre-treinamento.

    Args:
        sigspatial_dir: Diretorio raiz do dataset SIGSPATIAL.
        output_dir: Diretorio de saida (default: sigspatial_dir).
        max_negative_ratio: Maximo de tiles negativos por cada tile positivo.
    """
    if output_dir is None:
        output_dir = sigspatial_dir

    tiles_dir = output_dir / "tiles"
    masks_dir = output_dir / "masks"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Carregar poligonos de lagos
    gpkg_path = None
    for f in sigspatial_dir.rglob("lake_polygons_training.gpkg"):
        gpkg_path = f
        break

    if gpkg_path is None:
        raise FileNotFoundError("lake_polygons_training.gpkg nao encontrado!")

    print(f"Carregando poligonos de: {gpkg_path}")
    lakes_gdf = gpd.read_file(gpkg_path)
    print(f"  Total de poligonos de lagos: {len(lakes_gdf)}")

    # Agrupar por imagem
    images = lakes_gdf["image"].unique()
    print(f"  Imagens referenciadas: {len(images)}")

    total_tiles = 0
    total_positive = 0
    total_negative = 0

    for image_name in images:
        print(f"\n{'='*60}")
        print(f"Processando: {image_name}")
        print(f"{'='*60}")

        # Encontrar o TIF correspondente
        tif_path = find_tif_for_image(image_name, sigspatial_dir)
        if tif_path is None:
            print(f"  [AVISO] TIF nao encontrado para {image_name}, pulando...")
            continue
        print(f"  TIF: {tif_path.name}")

        # Filtrar lagos desta imagem
        image_lakes = lakes_gdf[lakes_gdf["image"] == image_name]
        print(f"  Lagos nesta imagem: {len(image_lakes)}")

        # Obter regioes de treinamento (bounding boxes dos lagos)
        # Para ser eficiente, so processar tiles perto dos lagos
        lake_bounds = image_lakes.total_bounds  # [minx, miny, maxx, maxy]

        with rasterio.open(tif_path) as src:
            print(f"  Raster: {src.height}x{src.width}, CRS={src.crs}")

            # Converter bounds dos lagos para pixel coords
            inv_transform = ~src.transform
            col_min, row_max = inv_transform * (lake_bounds[0], lake_bounds[1])
            col_max, row_min = inv_transform * (lake_bounds[2], lake_bounds[3])

            # Margem extra de 2 tiles ao redor da area dos lagos
            margin = TILE_SIZE * 2
            row_start = max(0, int(row_min) - margin)
            row_end = min(src.height, int(row_max) + margin)
            col_start = max(0, int(col_min) - margin)
            col_end = min(src.width, int(col_max) + margin)

            print(f"  Area de interesse: rows [{row_start}:{row_end}], cols [{col_start}:{col_end}]")
            n_rows = (row_end - row_start - TILE_SIZE) // STRIDE + 1
            n_cols = (col_end - col_start - TILE_SIZE) // STRIDE + 1
            print(f"  Grid estimado: {n_rows}x{n_cols} = ~{n_rows * n_cols} tiles")

            # Preparar geometrias para rasterizacao
            lake_geoms = list(image_lakes.geometry)

            image_positive = 0
            image_negative = 0
            image_stem = Path(image_name).stem

            for row_off in range(row_start, row_end - TILE_SIZE + 1, STRIDE):
                for col_off in range(col_start, col_end - TILE_SIZE + 1, STRIDE):
                    window = Window(col_off, row_off, TILE_SIZE, TILE_SIZE)

                    # Ler tile RGB
                    tile_data = src.read([1, 2, 3], window=window)  # (3, H, W)

                    # Verificar validade (nao muito preto/nodata)
                    valid_ratio = np.mean(tile_data.sum(axis=0) > 0)
                    if valid_ratio < MIN_VALID_RATIO:
                        continue

                    # Rasterizar lagos neste tile
                    tile_transform = rasterio.windows.transform(window, src.transform)
                    mask = rasterize(
                        lake_geoms,
                        out_shape=(TILE_SIZE, TILE_SIZE),
                        transform=tile_transform,
                        fill=0,
                        default_value=255,
                        dtype=np.uint8,
                    )

                    has_lake = mask.max() > 0

                    # Controle de balanceamento: limitar tiles negativos
                    if not has_lake:
                        if image_negative >= max_negative_ratio * max(image_positive, 1):
                            continue
                        image_negative += 1
                    else:
                        image_positive += 1

                    # Salvar tile e mascara
                    tile_id = f"{image_stem}_r{row_off}_c{col_off}"
                    tile_img = Image.fromarray(tile_data.transpose(1, 2, 0))  # HWC
                    mask_img = Image.fromarray(mask)

                    tile_img.save(tiles_dir / f"{tile_id}.png")
                    mask_img.save(masks_dir / f"{tile_id}.png")

            print(f"  Tiles positivos (com lago): {image_positive}")
            print(f"  Tiles negativos (sem lago): {image_negative}")
            total_positive += image_positive
            total_negative += image_negative
            total_tiles += image_positive + image_negative

    print(f"\n{'='*60}")
    print(f"RESULTADO FINAL")
    print(f"{'='*60}")
    print(f"Total de tiles: {total_tiles}")
    print(f"  Positivos (com lago): {total_positive} ({100*total_positive/max(total_tiles,1):.1f}%)")
    print(f"  Negativos (sem lago): {total_negative} ({100*total_negative/max(total_tiles,1):.1f}%)")
    print(f"\nSaida:")
    print(f"  Tiles: {tiles_dir}")
    print(f"  Masks: {masks_dir}")
    print(f"\nProximo passo:")
    print(f"  python 03a_pretrain_satellite.py --data-dir {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepara dataset SIGSPATIAL Cup 2023 para pre-treinamento SAM"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="data/SIGSPATIAL",
        help="Diretorio raiz do dataset SIGSPATIAL"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Diretorio de saida (default: mesmo que --data-dir)"
    )
    parser.add_argument(
        "--neg-ratio", type=float, default=1.5,
        help="Max tiles negativos por cada positivo (default: 1.5)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir

    prepare_tiles(data_dir, output_dir, max_negative_ratio=args.neg_ratio)


if __name__ == "__main__":
    main()
