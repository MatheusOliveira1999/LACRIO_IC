"""
convert_qgis_to_masks.py - Converte anotações vetoriais do QGIS em máscaras PNG

Projeto: LACRIO IC - Extração de Feições Supraglaciais
Data: Janeiro 2026

Uso:
    python convert_qgis_to_masks.py                     # Converte todas as feições
    python convert_qgis_to_masks.py --feature lakes    # Apenas lagos
    python convert_qgis_to_masks.py --year 2019        # Apenas ano específico
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from osgeo import gdal, ogr
    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False
    print("⚠️  GDAL/OGR não encontrado. Instale com: conda install gdal")

# Importar configurações
sys.path.insert(0, str(Path(__file__).parent))
from config import Config


def load_geopackage(gpkg_path: Path) -> dict:
    """
    Carrega um GeoPackage e retorna as geometrias por tile_id.
    
    Args:
        gpkg_path: Caminho para o arquivo .gpkg
        
    Returns:
        Dict mapeando tile_id -> lista de geometrias
    """
    if not gpkg_path.exists():
        print(f"⚠️  Arquivo não encontrado: {gpkg_path}")
        return {}
    
    datasource = ogr.Open(str(gpkg_path))
    if datasource is None:
        print(f"❌ Erro ao abrir: {gpkg_path}")
        return {}
    
    layer = datasource.GetLayer(0)
    geometries_by_tile = {}
    
    for feature in layer:
        tile_id = feature.GetField("tile_id")
        geom = feature.GetGeometryRef()
        
        if tile_id is not None and geom is not None:
            if tile_id not in geometries_by_tile:
                geometries_by_tile[tile_id] = []
            
            # Clonar geometria para evitar problemas de memória
            geometries_by_tile[tile_id].append(geom.Clone())
    
    datasource = None  # Fechar
    return geometries_by_tile


def geometry_to_mask(geometry, width: int = 512, height: int = 512) -> np.ndarray:
    """
    Converte uma geometria OGR em máscara numpy.
    
    Args:
        geometry: Geometria OGR (polígono ou linha)
        width: Largura da máscara
        height: Altura da máscara
        
    Returns:
        Máscara binária (0/255)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    geom_type = geometry.GetGeometryType()
    
    # Extrair pontos da geometria
    if geom_type in [ogr.wkbPolygon, ogr.wkbPolygon25D]:
        ring = geometry.GetGeometryRef(0)
        points = []
        for i in range(ring.GetPointCount()):
            x, y, _ = ring.GetPoint(i)
            points.append([int(x), int(y)])
        
        if len(points) > 2:
            pts = np.array([points], dtype=np.int32)
            cv2.fillPoly(mask, pts, 255)
    
    elif geom_type in [ogr.wkbLineString, ogr.wkbLineString25D]:
        points = []
        for i in range(geometry.GetPointCount()):
            x, y, _ = geometry.GetPoint(i)
            points.append([int(x), int(y)])
        
        if len(points) > 1:
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(mask, [pts], False, 255, thickness=3)
    
    elif geom_type in [ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D]:
        for i in range(geometry.GetGeometryCount()):
            sub_geom = geometry.GetGeometryRef(i)
            sub_mask = geometry_to_mask(sub_geom, width, height)
            mask = np.maximum(mask, sub_mask)
    
    return mask


def convert_feature(feature_name: str, year: int, qgis_dir: Path) -> int:
    """
    Converte anotações de uma feição para máscaras PNG.
    
    Args:
        feature_name: Nome da feição (lakes, crevasses, channels)
        year: Ano dos tiles
        qgis_dir: Diretório com arquivos QGIS
        
    Returns:
        Número de máscaras criadas
    """
    gpkg_path = qgis_dir / f"{feature_name}_annotations.gpkg"
    output_dir = Config.MASKS_DIR / str(year) / "annotations" / feature_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 Convertendo: {feature_name}")
    print(f"   Entrada: {gpkg_path}")
    print(f"   Saída: {output_dir}")
    
    geometries = load_geopackage(gpkg_path)
    
    if not geometries:
        print(f"   ⚠️  Nenhuma anotação encontrada")
        return 0
    
    count = 0
    for tile_id, geom_list in geometries.items():
        # Criar máscara combinando todas as geometrias do tile
        combined_mask = np.zeros((512, 512), dtype=np.uint8)
        
        for geom in geom_list:
            mask = geometry_to_mask(geom)
            combined_mask = np.maximum(combined_mask, mask)
        
        # Salvar
        mask_filename = f"tile_{tile_id:06d}_{feature_name}.png"
        mask_path = output_dir / mask_filename
        cv2.imwrite(str(mask_path), combined_mask)
        count += 1
    
    print(f"   ✓ Criadas {count} máscaras")
    return count


def update_annotations_index(year: int):
    """Atualiza o índice de anotações."""
    annotations_dir = Config.MASKS_DIR / str(year) / "annotations"
    index_path = annotations_dir / "annotations_index.json"
    
    annotations = {}
    
    for feature in Config.FEATURES.keys():
        feature_dir = annotations_dir / feature
        if feature_dir.exists():
            masks = list(feature_dir.glob("*.png"))
            tile_ids = []
            for mask_path in masks:
                # Extrair tile_id do nome: tile_000123_lakes.png
                parts = mask_path.stem.split("_")
                if len(parts) >= 2:
                    try:
                        tile_id = int(parts[1])
                        tile_ids.append(tile_id)
                    except ValueError:
                        pass
            annotations[feature] = sorted(tile_ids)
    
    # Salvar índice
    with open(index_path, "w") as f:
        json.dump(annotations, f, indent=2)
    
    print(f"\n✓ Índice atualizado: {index_path}")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Converte anotações QGIS em máscaras PNG"
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        default=2016,
        choices=Config.YEARS,
        help="Ano dos tiles"
    )
    parser.add_argument(
        "--feature", "-f",
        type=str,
        choices=list(Config.FEATURES.keys()) + ["all"],
        default="all",
        help="Feição a converter (ou 'all')"
    )
    parser.add_argument(
        "--qgis-dir", "-q",
        type=str,
        default=None,
        help="Diretório com arquivos QGIS (default: PROJECT_DIR/qgis)"
    )
    
    args = parser.parse_args()
    
    if not GDAL_AVAILABLE:
        print("❌ Erro: GDAL/OGR não instalado")
        print("   Execute: conda install gdal")
        sys.exit(1)
    
    # Diretório QGIS
    qgis_dir = Path(args.qgis_dir) if args.qgis_dir else Config.PROJECT_DIR / "qgis"
    
    if not qgis_dir.exists():
        print(f"⚠️  Diretório QGIS não existe: {qgis_dir}")
        print(f"   Criando diretório...")
        qgis_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Coloque os arquivos .gpkg neste diretório")
        sys.exit(0)
    
    print("=" * 60)
    print("CONVERSÃO DE ANOTAÇÕES QGIS → MÁSCARAS PNG")
    print("=" * 60)
    print(f"Ano: {args.year}")
    print(f"Diretório QGIS: {qgis_dir}")
    
    # Converter
    total = 0
    
    if args.feature == "all":
        features = list(Config.FEATURES.keys())
    else:
        features = [args.feature]
    
    for feature in features:
        count = convert_feature(feature, args.year, qgis_dir)
        total += count
    
    # Atualizar índice
    update_annotations_index(args.year)
    
    print("\n" + "=" * 60)
    print(f"✅ Conversão concluída: {total} máscaras criadas")
    print("=" * 60)


if __name__ == "__main__":
    main()
