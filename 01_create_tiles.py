"""
01_create_tiles.py - Divide mosaicos em tiles para processamento com SAM

Projeto: LACRIO IC - Extração de Feições Supraglaciais
Data: Janeiro 2026

Uso:
    python 01_create_tiles.py              # Processa todos os anos
    python 01_create_tiles.py --year 2019  # Processa apenas 2019
    python 01_create_tiles.py --info       # Mostra info dos mosaicos
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

# Importar configurações
sys.path.insert(0, str(Path(__file__).parent))
from config import Config


def get_mosaic_info(mosaic_path: Path) -> dict:
    """
    Retorna informações sobre um mosaico GeoTIFF.
    
    Args:
        mosaic_path: Caminho para o arquivo .tif
        
    Returns:
        Dicionário com metadados do mosaico
    """
    with rasterio.open(mosaic_path) as src:
        info = {
            "path": str(mosaic_path),
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": str(src.dtypes[0]),
            "crs": str(src.crs),
            "resolution": src.res,
            "bounds": src.bounds,
            "size_gb": mosaic_path.stat().st_size / (1024**3)
        }
        
        # Calcular número estimado de tiles
        step = Config.TILE_SIZE - Config.OVERLAP
        n_tiles_x = max(1, (src.width - Config.TILE_SIZE) // step + 1)
        n_tiles_y = max(1, (src.height - Config.TILE_SIZE) // step + 1)
        info["estimated_tiles"] = n_tiles_x * n_tiles_y
        
    return info


def get_mosaic_path_for_year(year: int, source_dir: Path | None = None) -> Path:
    """
    Resolve caminho do mosaico para um ano, com opção de sobrescrever diretório.

    Args:
        year: Ano do mosaico
        source_dir: Diretório opcional com mosaicos tratados

    Returns:
        Caminho para o mosaico
    """
    if source_dir is None:
        return Config.get_mosaic_path(year)

    if year not in Config.MOSAICS:
        raise ValueError(f"Ano {year} não disponível. Anos válidos: {Config.YEARS}")

    preferred = source_dir / Config.MOSAICS[year]
    if preferred.exists():
        return preferred

    patterns = [
        f"*{year}*.tif",
        f"*{year}*.TIF",
        f"*{year}*.tiff",
        f"*{year}*.TIFF",
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend([p for p in source_dir.glob(pattern) if p.is_file()])

    filtered = [p for p in candidates if "mosaic" in p.name.lower()]
    if filtered:
        return sorted(filtered, key=lambda p: p.name)[0]

    return preferred


def create_tiles(mosaic_path: Path, output_dir: Path, 
                 tile_size: int = 512, overlap: int = 64,
                 min_valid_ratio: float = 0.7) -> list:
    """
    Divide mosaico grande em tiles para SAM.
    
    Args:
        mosaic_path: Caminho para o mosaico GeoTIFF
        output_dir: Diretório de saída para tiles
        tile_size: Tamanho do tile em pixels
        overlap: Sobreposição entre tiles adjacentes
        min_valid_ratio: Mínimo de pixels válidos (não NoData)
        
    Returns:
        Lista com metadados dos tiles criados
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tiles_info = []
    
    with rasterio.open(mosaic_path) as src:
        print(f"\n📂 Mosaico: {mosaic_path.name}")
        print(f"   Dimensões: {src.width} x {src.height} pixels")
        print(f"   Bandas: {src.count}")
        print(f"   CRS: {src.crs}")
        print(f"   Resolução: {src.res[0]:.4f} m/pixel")
        
        tile_id = 0
        step = tile_size - overlap
        skipped = 0
        
        # Calcular número total de tiles para barra de progresso
        n_tiles_x = max(1, (src.width - tile_size) // step + 1)
        n_tiles_y = max(1, (src.height - tile_size) // step + 1)
        total_tiles = n_tiles_x * n_tiles_y
        
        print(f"   Tiles estimados: {total_tiles}")
        print(f"   Tamanho tile: {tile_size}x{tile_size} (overlap: {overlap})")
        
        with tqdm(total=total_tiles, desc="   Criando tiles", unit="tile") as pbar:
            for y in range(0, src.height - tile_size + 1, step):
                for x in range(0, src.width - tile_size + 1, step):
                    window = Window(x, y, tile_size, tile_size)
                    
                    # Ler RGB (assumindo bandas 1, 2, 3)
                    try:
                        if src.count >= 3:
                            rgb = src.read([1, 2, 3], window=window)
                        else:
                            # Se monocromático, replicar para 3 canais
                            gray = src.read(1, window=window)
                            rgb = np.stack([gray, gray, gray])
                    except Exception as e:
                        pbar.update(1)
                        skipped += 1
                        continue
                    
                    # Verificar se tile tem dados válidos
                    # NoData geralmente é 0 ou valores muito baixos
                    valid_pixels = np.sum(rgb > 0)
                    total_pixels = tile_size * tile_size * 3
                    valid_ratio = valid_pixels / total_pixels
                    
                    if valid_ratio < min_valid_ratio:
                        pbar.update(1)
                        skipped += 1
                        continue
                    
                    # Converter para formato HWC (Height, Width, Channels)
                    rgb_hwc = np.transpose(rgb, (1, 2, 0))
                    
                    # Normalizar para 8-bit se necessário
                    if rgb_hwc.dtype != np.uint8:
                        # Assumir 16-bit ou float, normalizar para 0-255
                        rgb_min = rgb_hwc.min()
                        rgb_max = rgb_hwc.max()
                        if rgb_max > rgb_min:
                            rgb_hwc = ((rgb_hwc - rgb_min) / (rgb_max - rgb_min) * 255).astype(np.uint8)
                        else:
                            rgb_hwc = np.zeros_like(rgb_hwc, dtype=np.uint8)
                    
                    # Salvar como PNG (SAM espera RGB padrão)
                    tile_filename = f"tile_{tile_id:06d}.png"
                    tile_path = output_dir / tile_filename
                    
                    # OpenCV espera BGR
                    cv2.imwrite(str(tile_path), cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR))
                    
                    # Guardar metadados para reconstrução posterior
                    transform = rasterio.windows.transform(window, src.transform)
                    tiles_info.append({
                        "id": tile_id,
                        "filename": tile_filename,
                        "x": x,
                        "y": y,
                        "width": tile_size,
                        "height": tile_size,
                        "transform": list(transform)[:6],
                        "valid_ratio": float(valid_ratio)
                    })
                    
                    tile_id += 1
                    pbar.update(1)
            
            # Atualizar barra para tiles restantes (pulados)
            remaining = total_tiles - pbar.n
            if remaining > 0:
                pbar.update(remaining)
        
        # Salvar índice de tiles
        index_path = output_dir / "tiles_index.json"
        with open(index_path, "w") as f:
            json.dump({
                "source": str(mosaic_path),
                "tile_size": tile_size,
                "overlap": overlap,
                "total_tiles": tile_id,
                "skipped_tiles": skipped,
                "crs": str(src.crs),
                "original_width": src.width,
                "original_height": src.height,
                "tiles": tiles_info
            }, f, indent=2)
        
        print(f"\n✓ Criados {tile_id} tiles ({skipped} pulados por NoData)")
        print(f"✓ Índice salvo: {index_path}")
        
        return tiles_info


def process_year(year: int, source_dir: Path | None = None) -> int:
    """
    Processa mosaico de um ano específico.
    
    Args:
        year: Ano a processar
        
    Returns:
        Número de tiles criados
    """
    mosaic_path = get_mosaic_path_for_year(year, source_dir=source_dir)
    
    if not mosaic_path.exists():
        print(f"⚠️  Mosaico {year} não encontrado: {mosaic_path}")
        return 0
    
    output_dir = Config.TILES_DIR / str(year)
    
    tiles = create_tiles(
        mosaic_path=mosaic_path,
        output_dir=output_dir,
        tile_size=Config.TILE_SIZE,
        overlap=Config.OVERLAP,
        min_valid_ratio=Config.MIN_VALID_RATIO
    )
    
    return len(tiles)


def process_all_years(source_dir: Path | None = None) -> dict:
    """
    Processa todos os anos disponíveis.
    
    Returns:
        Dicionário com número de tiles por ano
    """
    Config.create_directories()
    
    results = {}
    total_tiles = 0
    
    for year in Config.YEARS:
        print(f"\n{'='*60}")
        print(f"ANO {year}")
        print(f"{'='*60}")
        
        n_tiles = process_year(year, source_dir=source_dir)
        results[year] = n_tiles
        total_tiles += n_tiles
    
    print(f"\n{'='*60}")
    print("RESUMO")
    print(f"{'='*60}")
    for year, count in results.items():
        print(f"  {year}: {count:,} tiles")
    print(f"  TOTAL: {total_tiles:,} tiles")
    
    return results


def show_mosaics_info(source_dir: Path | None = None):
    """Mostra informações sobre todos os mosaicos disponíveis."""
    print("\n" + "="*60)
    print("INFORMAÇÕES DOS MOSAICOS")
    print("="*60)
    
    for year in Config.YEARS:
        mosaic_path = get_mosaic_path_for_year(year, source_dir=source_dir)
        
        if not mosaic_path.exists():
            print(f"\n{year}: ⚠️  Não encontrado")
            continue
        
        info = get_mosaic_info(mosaic_path)
        
        print(f"\n{year}:")
        print(f"  Arquivo: {mosaic_path.name}")
        print(f"  Tamanho: {info['size_gb']:.2f} GB")
        print(f"  Dimensões: {info['width']:,} x {info['height']:,} pixels")
        print(f"  Bandas: {info['bands']}")
        print(f"  Tiles estimados: {info['estimated_tiles']:,}")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Cria tiles a partir dos mosaicos para processamento com SAM"
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        choices=Config.YEARS,
        help=f"Ano específico para processar. Opções: {Config.YEARS}"
    )
    parser.add_argument(
        "--info", "-i",
        action="store_true",
        help="Apenas mostra informações dos mosaicos sem processar"
    )
    parser.add_argument(
        "--tile-size", "-t",
        type=int,
        default=Config.TILE_SIZE,
        help=f"Tamanho do tile (default: {Config.TILE_SIZE})"
    )
    parser.add_argument(
        "--overlap", "-o",
        type=int,
        default=Config.OVERLAP,
        help=f"Overlap entre tiles (default: {Config.OVERLAP})"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Diretório alternativo com mosaicos (ex.: mosaicos pré-processados)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretório de saída alternativo para os tiles (default: Config.TILES_DIR)"
    )

    args = parser.parse_args()

    # Atualizar config se parâmetros fornecidos
    if args.tile_size != Config.TILE_SIZE:
        Config.TILE_SIZE = args.tile_size
    if args.overlap != Config.OVERLAP:
        Config.OVERLAP = args.overlap
    if args.output_dir is not None:
        Config.TILES_DIR = args.output_dir.resolve()
        print(f"📁 Tiles serão salvos em: {Config.TILES_DIR}")
    
    source_dir = None
    if args.source_dir is not None:
        source_dir = args.source_dir.resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"Diretório não encontrado: {source_dir}")
        print(f"📁 Usando mosaicos de: {source_dir}")

    if args.info:
        show_mosaics_info(source_dir=source_dir)
        return
    
    Config.print_info()
    
    if args.year:
        Config.create_directories()
        process_year(args.year, source_dir=source_dir)
    else:
        process_all_years(source_dir=source_dir)
    
    print("\n✅ Processamento concluído!")


if __name__ == "__main__":
    main()
