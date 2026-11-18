"""
empacota_dados_revisor.py - Gera o pacote de dados para um revisor externo.

O repositorio no GitHub so tem codigo (tiles, mascaras e modelos nao sao
versionados). Este script junta o minimo de dados que o `annotate.py --review`
precisa: as mascaras anotadas de fendas (2016-2019) e os tiles correspondentes.

Uso:
    python empacota_dados_revisor.py            # -> backups/LACRIO_IC_dados_revisor.zip

O revisor extrai o zip DENTRO da pasta do projeto clonado (ele ja traz as
pastas tiles/ e masks/ na estrutura certa).
"""

import zipfile
from pathlib import Path

from config import Config

FEATURE = "crevasses"
YEARS = [2016, 2017, 2018, 2019]
OUT = Config.PROJECT_DIR / "backups" / "LACRIO_IC_dados_revisor.zip"


def main():
    OUT.parent.mkdir(exist_ok=True)
    n_mask = n_tile = 0
    faltando = []
    # PNG ja e comprimido: ZIP_STORED evita gastar tempo sem ganho
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
        for yr in YEARS:
            gt_dir = Config.MASKS_DIR / str(yr) / "annotations" / FEATURE
            idx = Config.TILES_DIR / str(yr) / "tiles_index.json"
            if idx.exists():
                z.write(idx, idx.relative_to(Config.PROJECT_DIR))
            for m in sorted(gt_dir.glob(f"tile_*_{FEATURE}.png")):
                tile = Config.TILES_DIR / str(yr) / f"{m.stem.replace(f'_{FEATURE}', '')}.png"
                if not tile.exists():
                    faltando.append(str(tile))
                    continue
                z.write(m, m.relative_to(Config.PROJECT_DIR))
                z.write(tile, tile.relative_to(Config.PROJECT_DIR))
                n_mask += 1
                n_tile += 1
    print(f"{OUT}  ({OUT.stat().st_size / 1e6:.0f} MB)")
    print(f"  {n_mask} mascaras + {n_tile} tiles ({', '.join(map(str, YEARS))})")
    if faltando:
        print(f"  AVISO: {len(faltando)} tiles sem imagem: {faltando[:5]}")


if __name__ == "__main__":
    main()
