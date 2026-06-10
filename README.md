# LACRIO IC — Extração de Feições Supraglaciais

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Projeto de Iniciação Científica** — análise de mudanças na superfície de geleiras com dados de VANT (drone) e deep learning.

> **Orientador:** Prof. Dr. Jorge Arigony Neto
> **Laboratório:** LaCrio — Monitoramento da Criosfera (FURG)
> **Bolsa:** CNPq — Vigência Set/2025 – Ago/2026

---

## Fase atual (mai/2026): foco em fendas no ano de 2016 — ✅ VALIDADA

Esta fase tem **escopo deliberadamente reduzido**:

1. **Trabalhar apenas com `crevasses` (fendas).** Lakes e channels ficam fora do ciclo.
2. **Treinar exclusivamente com dados de 2016** — ano de referência.
3. **Generalização temporal:** usar o modelo bem treinado em 2016 para **inferir nos demais anos (2017–2020)** sem re-treinar.

### Resultado (27/05/2026)

Modelo U-Net **6-canal** (RGB + relief + slope + curvature do DEM), treinado SOMENTE em 2016 com 42 tiles GT:

| Ano | F1 | IoU | Precision | Recall |
|-----|----:|----:|----:|----:|
| 2016 (treino) | **0.6356** | 0.4658 | 0.5417 | 0.7688 |
| 2017 (cross-year) | **0.8318** | 0.7120 | 0.7552 | 0.9256 |
| 2018 (cross-year) | **0.8746** | 0.7771 | 0.8253 | 0.9301 |
| 2019 (cross-year) | **0.8369** | 0.7196 | 0.7524 | 0.9429 |

**Supera baseline anterior** (modelo de 4 anos: F1 0.71–0.78 em cada ano) treinando em **1 ano em 23 min**.

---

## Pipeline U-Net 6-canal (modelo de produção)

A versão SAM foi descontinuada em abr/2026 (gap treino-inferência inerente). Scripts SAM em [`archive/`](archive/).

**Inovações principais (mai/2026):**
- **DEM como canais de input** (`--use-dem-channels`): U-Net com `in_channels=6` recebe RGB + relief + slope + curvature normalizados. Topografia ajuda o modelo a discriminar fenda (depressão real) de detrito escuro (superfície plana).
- **Augmentation cromática direcionada**: `_gradient_brightness` custom simula iluminação heterogênea de drone que voou em horários diferentes. Combinado com HueSaturationValue + RGBShift.
- **Filtro DEM pós-inferência opcional** (`--dem-filter`): para usar com checkpoints 3-canal antigos.

```
Mosaico GeoTIFF
    ↓ 01_create_tiles.py
Tiles 512×512 PNG
    ↓ annotate.py  (pincel + zoom + modo revisão)
Máscaras GT (masks/{ano}/annotations/{feicao}/)
    ↓ 03_train_unet.py  (U-Net ResNet34 + Tversky + discriminative LR)
Modelo treinado (models/unet_{feicao}_best.pth)
    ↓ 04_inference_unet.py
Máscaras preditas (masks/{ano}/{feicao}/)
    ↓ 05_reconstruct_mosaic.py
Mosaico georreferenciado (results/)
    ↓ 06_validate.py
Métricas F1/IoU/Precision/Recall
```

---

## Área de estudo

**Glaciar Schiaparelli** — Cordilheira Darwin, Terra do Fogo, Chile.

| Dados | Período | Resolução |
|-------|---------|-----------|
| Mosaicos RGB (VANT) | 2016–2020 | 5,4 cm/px |
| DEMs | 2016–2022 | 22 cm/px |

---

## Estrutura do projeto

```
LACRIO_IC/
├── config.py                  # Configurações centralizadas
├── 01_create_tiles.py         # Gerar tiles 512×512 a partir dos mosaicos
├── 03_train_unet.py           # Treinar U-Net ResNet34
├── 04_inference_unet.py       # Inferência U-Net (forward pass único)
├── 05_reconstruct_mosaic.py   # Reconstruir mosaico georreferenciado
├── 06_validate.py             # F1, IoU, precision, recall
├── annotate.py                # Anotação manual (pincel + zoom + revisão)
├── active_learning.py         # Seleção de tiles para anotar (strategy random)
├── check_empty_masks.py       # Limpa máscaras com 0 pixels (bug histórico)
├── shadow_utils.py            # Hillshade DEM para filtro de sombra
├── generate_report_pdf.py     # Gera relatório PDF
├── archive/                   # Scripts SAM-era (descontinuados em abr/2026)
├── docs/                      # Plano C-TransUNet + paper + archive/
├── Schiaparelli_glacier/      # Mosaicos + DEMs (não versionado)
├── tiles/                     # Tiles gerados
├── masks/                     # Anotações + predições
├── models/                    # Checkpoints .pth
└── results/                   # Mosaicos finais + métricas
```

---

## Instalação

```bash
git clone https://github.com/MatheusOliveira1999/LACRIO_IC.git
cd LACRIO_IC

conda create -n sam_glaciar python=3.10 -y
conda activate sam_glaciar
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

---

## Uso — fluxo da fase atual (crevasses 2016)

```bash
# 1. Gerar tiles (se ainda não gerados)
python 01_create_tiles.py --year 2016

# 2. Anotar (ou revisar) fendas em 2016
python annotate.py --feature crevasses --year 2016 --review

# 3. Treinar U-Net 6-canal em 2016 (config oficial)
python 03_train_unet.py --feature crevasses --years 2016 \
    --loss bce_tversky --fp-weight 0.7 --fn-weight 0.3 \
    --epochs 200 --patience 100 --use-dem-channels

# 4. Inferir e validar no próprio 2016 (sanity check)
python 04_inference_unet.py --feature crevasses --year 2016 \
    --annotated-only --threshold 0.5 --no-feature-filter --validate

# 5. Transferência temporal: aplicar o mesmo modelo nos demais anos
#    (DEM features são carregadas automaticamente do ano de inferência)
for YEAR in 2017 2018 2019 2020; do
    python 04_inference_unet.py --feature crevasses --year $YEAR \
        --threshold 0.5 --no-feature-filter
done

# 6. Reconstruir mosaicos georreferenciados por ano
for YEAR in 2017 2018 2019 2020; do
    python 05_reconstruct_mosaic.py --feature crevasses --year $YEAR
done

# 7. (Opcional) Validar quantitativamente em anos com GT anotado
for YEAR in 2017 2018 2019; do
    python 04_inference_unet.py --feature crevasses --year $YEAR \
        --annotated-only --threshold 0.5 --no-feature-filter --validate
done
```

---

## Estado da arte (referência)

| Método | Feição | Resolução | F1 / IoU |
|--------|--------|-----------|----------|
| Wallace et al. (2025) — SAM 2 zero-shot | Crevasses | drone | IoU 0,28 |
| Chai et al. (2025) — SAM adapted | Lagos | Sentinel-2 | F1 0,88 |
| Este projeto (U-Net 4 anos, 04/05/2026) | Crevasses | drone 5,4 cm/px | F1 0,71–0,78 (2017–2019) |
| **Este projeto (U-Net 6-canal só 2016, 27/05/2026)** | Crevasses | drone 5,4 cm/px | **F1 0,83–0,87 (2017–2019)** · IoU 0,71–0,78 |

*Caveat: comparação com Wallace 2025 não é direta — zero-shot vs fine-tuned, domínios geográficos diferentes (Svalbard vs Patagônia), critério de anotação não comparável.*

---

## Documentação

- `docs/PLANO_IMPLEMENTACAO_ARTIGO_CTRANSUNET_3D.md` — roadmap futuro (geometria 2D/3D, strain)
- `docs/archive/` — relatórios e diagnósticos da fase SAM
- **Vault Obsidian** (`~/Documents/obsidian/LACRIO/`) — progresso, decisões técnicas e definições estritas de cada feição

---

## Licença

MIT.

## Contato

**Matheus Oliveira** — Oceanologia FURG · GitHub: [@MatheusOliveira1999](https://github.com/MatheusOliveira1999)
