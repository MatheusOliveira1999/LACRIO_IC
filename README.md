# LACRIO IC — Extração de Feições Supraglaciais

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Projeto de Iniciação Científica** — análise de mudanças na superfície de geleiras com dados de VANT (drone) e deep learning.

> **Orientador:** Prof. Dr. Jorge Arigony Neto
> **Laboratório:** LaCrio — Monitoramento da Criosfera (FURG)
> **Bolsa:** CNPq — Vigência Set/2025 – Ago/2026

---

## Estado atual (set/2026): fendas, treino em 2016, teste em blocos nunca vistos

Escopo: apenas `crevasses` (fendas); treino **somente com 2016**; o mesmo modelo é aplicado a 2017–2019 sem re-treino.

Modelo U-Net **6 canais** (RGB + relief + slope + curvature do DEM). Gabarito revisado de 2016: **218 recortes** (144 com fenda + 74 conferidos sem fenda).

| Teste | Precisão | Revocação | F1 |
|-------|----:|----:|----:|
| **2016, validação cruzada espacial 5-fold** (blocos nunca vistos) | 0,627 | 0,703 | **0,661 ± 0,018** |
| 2016, segunda divisão dos blocos | — | — | 0,663 ± 0,030 |
| 2016, com folga de 4 px (32 cm) na borda | 0,836 | 0,856 | 0,846 |
| 2017 / 2018 / 2019, um modelo (cross-year) | 0,67 | 0,66–0,75 | 0,679 / 0,708 / 0,664 |
| 2017 / 2018 / 2019, comitê de 10 modelos | 0,65–0,68 | 0,71–0,80 | 0,689 / 0,714 / 0,702 |

> **Nota:** versões anteriores deste README traziam F1 de 0,83–0,87 nos anos 2017–2019. Esses valores eram otimistas (a avaliação era circular) e foram substituídos pelos números acima. Detalhes em [`relatorio_final/relatorio.md`](relatorio_final/relatorio.md).

O principal limite hoje é o gabarito, não a rede: a auto-concordância do anotador (re-anotação às cegas de 15 recortes) é F1 0,583 estrito / 0,785 com folga de 4 px. Por isso a etapa em andamento é a **revisão das anotações por um segundo revisor** (seção abaixo).

---

## Guia do revisor de anotações

Para revisar as máscaras de fendas **não é preciso GPU nem PyTorch** — só Python e três bibliotecas leves.

**1. Instalar o Python 3.10 ou mais novo** ([python.org](https://www.python.org/downloads/); no Windows, marcar *Add Python to PATH*).

**2. Baixar o código** — `git clone https://github.com/MatheusOliveira1999/LACRIO_IC.git`, ou botão verde *Code → Download ZIP* e extrair.

**3. Baixar os dados** — as imagens e máscaras não ficam no GitHub. O arquivo `LACRIO_IC_dados_revisor.zip` (~120 MB; 335 recortes anotados de 2016–2019 + máscaras) é enviado à parte pelo Matheus. **Extraia-o dentro da pasta do projeto**: devem aparecer as pastas `tiles/` e `masks/` ao lado do `annotate.py`.

**4. Instalar as dependências** (terminal aberto na pasta do projeto):

```bash
pip install -r requirements-revisor.txt
```

**5. Abrir o anotador em modo revisão** (um ano por vez; 2016 tem 218 recortes, os demais ~40 cada):

```bash
python annotate.py --feature crevasses --year 2016 --review --no-model
```

A máscara atual aparece em vermelho sobre a imagem. Corrija o que for preciso e avance.

| Ação | Controle |
|------|----------|
| Pintar fenda | botão esquerdo |
| Apagar | botão direito |
| Tamanho do pincel | scroll (ou `+` / `-`) |
| Zoom / mover / resetar | `Ctrl`+scroll / botão do meio + arrastar / `0` |
| Alternar modo linha/pixel | `L` |
| Desfazer última pincelada | `Z` |
| **Salvar e ir ao próximo** | `Enter` ou `Espaço` |
| Pular sem salvar | `S` |
| Marcar "sem fenda" (máscara vazia) | `R` |
| Sair | `Q` ou `Esc` |

O trabalho pode ser feito em várias sessões: tudo que foi salvo com `Enter` fica gravado em `masks/`.

**6. Devolver** — compactar a pasta `masks/` e enviar ao Matheus.

*(Para o Matheus: o pacote de dados é gerado por `python empacota_dados_revisor.py`.)*

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
├── 10_crossval_curves.py      # Validação cruzada espacial + curvas PR/ROC
├── 13_intra_anotador.py       # Consistência intra-anotador (re-anotação às cegas)
├── 16_oof_ensemble_eval.py    # Avaliação do comitê (ensemble) fora da amostra
├── 17_compare_oof.py          # Comparação pareada entre versões (bootstrap + Wilcoxon)
├── empacota_dados_revisor.py  # Gera o zip de dados para o revisor externo
├── shadow_utils.py            # Hillshade DEM para filtro de sombra
├── generate_report_pdf.py     # Gera relatório PDF
├── archive/                   # Scripts SAM-era (descontinuados em abr/2026)
├── report_figs/               # Scripts das figuras do relatório
├── relatorio_final/           # Relatório final (fonte .md + scripts de build)
├── Schiaparelli_glacier/      # Mosaicos + DEMs (não versionado)
├── tiles/                     # Tiles gerados
├── masks/                     # Anotações + predições
├── models/                    # Checkpoints .pth
└── results/                   # Mosaicos finais + métricas
```

---

## Instalação (pipeline completo)

Para apenas revisar anotações, ver o [Guia do revisor](#guia-do-revisor-de-anotações) — não precisa de nada disto.

```bash
git clone https://github.com/MatheusOliveira1999/LACRIO_IC.git
cd LACRIO_IC

conda create -n lacrio python=3.10 -y
conda activate lacrio
# PyTorch: usar o comando para a sua GPU/CPU em https://pytorch.org
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
| **Este projeto (U-Net 6 canais, treino só em 2016, set/2026)** | Crevasses | drone, 8 cm/px | **F1 0,661 ± 0,018** (CV espacial) · 0,69–0,71 em 2017–2019 (comitê) |

*Caveat: comparação com Wallace 2025 não é direta — zero-shot vs fine-tuned, domínios geográficos diferentes (Svalbard vs Patagônia), critério de anotação não comparável.*

---

## Documentação

- [`relatorio_final/relatorio.md`](relatorio_final/relatorio.md) — relatório final da IC (método, resultados, limites)
- **Vault Obsidian** (`~/Documents/obsidian/LACRIO/`) — progresso, decisões técnicas e definições estritas de cada feição

---

## Licença

MIT.

## Contato

**Matheus Oliveira** — Oceanologia FURG · GitHub: [@MatheusOliveira1999](https://github.com/MatheusOliveira1999)
