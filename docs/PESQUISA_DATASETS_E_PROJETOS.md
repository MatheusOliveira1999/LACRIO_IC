# Pesquisa: Datasets, Projetos e Papers para Melhorar o Pipeline SAM

**Data:** 05 de Abril de 2026  
**Objetivo:** Identificar datasets, repositorios GitHub e tecnicas publicadas que possam melhorar a segmentacao de feicoes supraglaciais (lagos, crevasses, canais) no Glaciar Schiaparelli com SAM-HQ.

---

## 1. Paper Mais Relevante: SAM para Crevasses em Drone (Svalbard)

> **Wallace et al. (2025)** — "Exploring Segment Anything Foundation Models for Out of Domain Crevasse Drone Image Segmentation"  
> *Northern Lights Deep Learning Conference (NLDL), PMLR 265:255-268*

| Campo | Detalhe |
|-------|---------|
| **O que faz** | Avalia SAM e SAM 2 para segmentar crevasses em imagens de drone de geleiras |
| **Dados** | 10 imagens UAV de alta resolucao de Svalbard, Noruega |
| **Modelo** | SAM 2 Hiera-L (melhor resultado) |
| **Metricas** | DSC = 0.43, IoU = 0.28 (prompting automatico) |
| **Conclusao** | SAM nao funciona bem out-of-the-box; fine-tuning e necessario para domain shift |
| **Link** | [Paper](https://proceedings.mlr.press/v265/wallace25a.html) |

**Por que importa:** Este paper e o mais proximo do nosso projeto — mesmo dominio (drone glacial), mesma arquitetura (SAM), mesma feicao (crevasses). Os resultados deles (IoU 0.28) sao comparaveis aos nossos (IoU 0.16), confirmando que o domain shift drone-glacial e um desafio real. Eles sugerem few-shot learning e fine-tuning como caminhos de melhoria.

---

## 2. SAM Fine-Tuned para Lagos Supraglaciais (Groelandia)

> **Chai et al. (2025)** — "Potential of an adapting SAM for automatically extracting supraglacial lakes from satellite imagery over the Greenland ice sheet"  
> *International Journal of Digital Earth*

| Campo | Detalhe |
|-------|---------|
| **O que faz** | Fine-tune do SAM para extrair lagos supraglaciais na Groelandia |
| **Dados** | Sentinel-2 (10 m) e Landsat-8, treino na bacia SW da Groelandia |
| **Modelo** | SAM adaptado (adapting SAM) |
| **Metricas** | **F1 = 87.77%** (media), com apenas 20 amostras: F1 = 85.73% |
| **Ganho** | +19.18% sobre U-Net e DeepLabV3+ |
| **Link** | [Paper](https://www.tandfonline.com/doi/full/10.1080/17538947.2025.2554312) |

**Por que importa:** Demonstra que SAM fine-tuned com **pouquissimas amostras (20!)** atinge F1 de 85%+ para lagos. A diferenca e que eles usam satelite (10 m) e nos usamos drone (5.4 cm) — domain gap diferente, mas a estrategia de fine-tuning e diretamente aplicavel.

---

## 3. SAM em Glaciologia Multi-plataforma

> **Muchu et al. (2024)** — "Semantic segmentation of glaciological features across multiple remote sensing platforms with the Segment Anything Model (SAM)"  
> *Journal of Glaciology, Vol. 70*

| Campo | Detalhe |
|-------|---------|
| **O que faz** | Avalia SAM para segmentar feicoes glaciais em multiplas plataformas |
| **Dados** | Sentinel-1 SAR, PlanetScope (3 m), Landsat, timelapse cameras |
| **Feicoes** | Crevasses, icebergs, lagos supraglaciais, terminus glacial |
| **Resultado** | Com prompts manuais: resultados muito bons; sem prompts: limitado |
| **Anotacoes** | Ground truth criado com V7 Labs Darwin + iPad Pro |
| **Link** | [Paper](https://www.cambridge.org/core/journals/journal-of-glaciology/article/66D3A237ACB0975C9EE9BE19E0C2564E) |

**Por que importa:** Primeiro paper a testar SAM sistematicamente em feicoes glaciologicas. Confirma que SAM com prompts funciona bem, mas precisa de adaptacao para cada plataforma.

---

## 4. Deep Learning + SkySat para Hidrologia Supraglacial

> **Ryan et al. (2026)** — "Mechanisms of Surface Meltwater Ponding and Drainage on the Greenland Ice Sheet Revealed Using SkySat Imagery and Deep Learning"  
> *AGU Advances, Vol. 7*

| Campo | Detalhe |
|-------|---------|
| **O que faz** | Classifica agua superficial no Greenland com U-Net |
| **Dados** | SkySat (~1 m/pixel) — mais proximo de drone que Sentinel/Landsat |
| **Feicoes** | Lagos supraglaciais, canais supraglaciais, poças |
| **Descoberta** | Features pequenas (<1000 m²) representam 64% da area de agua |
| **Link** | [Paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025AV002030) |

**Por que importa:** Resolucao de ~1 m e mais proxima da nossa (5.4 cm) que satelites tradicionais. Demonstra que U-Net funciona bem para mapear a rede hidrografica supraglacial completa (lagos + canais). Poderia servir como baseline de comparacao.

---

## 5. Datasets Publicos Disponiveis

### 5.1 Datasets com Anotacoes (prontos para uso)

| Dataset | Feicoes | Resolucao | Formato | Link |
|---------|---------|-----------|---------|------|
| **SIGSPATIAL Cup 2023** — Lagos supraglaciais Groelandia | Lagos, cracks, moulins | Satelite (~3 m) | Tiles 1024x1024 + mascaras | [GitHub](https://github.com/knowledge-computing/sigspatial-cup-2023) |
| **Lutz et al. (2024)** — Lagos NE Groelandia 2016-2022 | Lagos supraglaciais | 10 m (Sentinel-2) | Shapefiles (poligonos) | [PANGAEA](https://doi.pangaea.de/10.1594/PANGAEA.973251) |
| **CALFIN** — Frentes de calvagem Groelandia 1972-2019 | Terminus glacial | 30 m (Landsat) + SAR | 22.678 frentes anotadas | [GitHub](https://github.com/daniel-cheng/CALFIN) |
| **NASA-IMPACT** — Lagos supraglaciais | Lagos (SGLs) | ~3 m (PlanetScope) | Tiles 96x96 + mascaras | [GitHub](https://github.com/NASA-IMPACT/veda-ai-supraglacial_segmentation) |
| **Calving Fronts and Where to Find Them** | Terminus glacial | Multi-sensor | Benchmark multi-modelo | [GitHub](https://github.com/Nora-Go/Calving_Fronts_and_Where_to_Find_Them) |

### 5.2 Ortomosaicos de Drone (sem anotacoes, mas com potencial)

| Dataset | Local | Resolucao | Conteudo | Link |
|---------|-------|-----------|----------|------|
| **Borebreen Glacier** (Zenodo, 2024) | Svalbard, Noruega | ~5-10 cm | Ortomosaico + DEM + modelo 3D | [Zenodo](https://zenodo.org/records/13837315) |
| **Bayelva Basin** (PANGAEA, 2024) | Svalbard, Noruega | ~5-10 cm | DSM, RGB, NIR, TIR | [PANGAEA](https://doi.pangaea.de/10.1594/PANGAEA.972779) |

### 5.3 Avaliacao de Utilidade para o Nosso Projeto

| Dataset | Util para treino direto? | Por que? |
|---------|-------------------------|----------|
| SIGSPATIAL Cup 2023 | **Parcial** | Lagos supraglaciais com mascaras, mas resolucao de satelite (3 m vs nossos 5.4 cm). Util para pre-treino |
| Lutz et al. PANGAEA | **Parcial** | Poligonos de lagos, mas Sentinel-2 (10 m). Bom para validacao cruzada |
| CALFIN | **Nao diretamente** | Frentes de calvagem, nao feicoes supraglaciais. Mas tecnicas de augmentation sao transferiveis |
| NASA-IMPACT | **Parcial** | Lagos com mascaras em ~3 m. Pode servir para pre-treino do decoder |
| Borebreen drone | **Sim (se anotar)** | Resolucao similar a nossa. Sem mascaras — precisaria anotar com nosso 02_sam_interactive.py |
| Wallace (Svalbard) | **Verificar** | 10 imagens de drone com crevasses — se os autores publicarem os dados, seria o dataset mais relevante |

---

## 6. Repositorios GitHub Mais Relevantes

### 6.1 Tier 1 — Diretamente aplicaveis ao nosso pipeline

| Repositorio | Descricao | Tecnica util | Stars | Link |
|-------------|-----------|-------------|-------|------|
| **segment-geospatial (samgeo)** | Pacote Python para SAM com suporte GeoTIFF | Tiling, reconstrucao de mosaicos, exportacao para vetor | ~3000+ | [GitHub](https://github.com/opengeos/segment-geospatial) |
| **Geo-SAM** | Plugin QGIS para anotacao interativa com SAM | Acelerar anotacoes direto no QGIS | ~200+ | [GitHub](https://github.com/coolzhao/Geo-SAM) |
| **SAM-Adapter-PyTorch** | Adapta SAM com adapter layers leves no encoder | Fine-tuning eficiente para domain shift grande | ~500+ | [GitHub](https://github.com/tianrun-chen/SAM-Adapter-PyTorch) |
| **MedSAM** | Fine-tuning de SAM para imagens medicas | Receita padrao-ouro: freeze encoder + treinar decoder | ~2000+ | [GitHub](https://github.com/bowang-lab/MedSAM) |
| **RSPrompter** | Geracao automatica de prompts para SAM em sensoriamento remoto | Elimina necessidade de prompts manuais na inferencia | ~300+ | [GitHub](https://github.com/KyanChen/RSPrompter) |

### 6.2 Tier 2 — Conhecimento glaciologico

| Repositorio | Descricao | Tecnica util | Link |
|-------------|-----------|-------------|------|
| **CALFIN** | Deteccao de frentes de calvagem com DeepLabV3+ | Augmentation pesada, distance-transform loss, pos-processamento | [GitHub](https://github.com/daniel-cheng/CALFIN) |
| **GlacierNet2** | Segmentacao multi-tarefa de geleiras | Multi-task head (debris + gelo + frente) | [GitHub](https://github.com/krismannino/GlacierNet2) |
| **glacier_mapping** | Mapeamento de geleiras com U-Net (Microsoft AI for Earth) | Class-balanced sampling, transfer learning | [GitHub](https://github.com/krisrs1128/glacier_mapping) |
| **NASA-IMPACT supraglacial** | Segmentacao de lagos supraglaciais | CNN + Operation IceBridge labels | [GitHub](https://github.com/NASA-IMPACT/veda-ai-supraglacial_segmentation) |
| **SIGSPATIAL Cup 2023** | Competicao de deteccao de lagos (SAM + DeepLabV3+) | Fine-tune SAM para lagos, benchmark | [GitHub](https://github.com/knowledge-computing/sigspatial-cup-2023) |

### 6.3 Tier 3 — Tecnicas uteis

| Repositorio | Tecnica | Aplicacao no nosso projeto | Link |
|-------------|---------|---------------------------|------|
| **Grounded-Segment-Anything** | Prompts de texto → bounding boxes → SAM | Prompt automatico: "supraglacial lake", "crevasse" | [GitHub](https://github.com/IDEA-Research/Grounded-Segment-Anything) |
| **MobileSAM / FastSAM** | SAM destilado, 10-50x mais rapido | Acelerar inferencia nos ~22k tiles | [GitHub MobileSAM](https://github.com/ChaoningZhang/MobileSAM) |
| **segmentation_models.pytorch** | U-Net, DeepLabV3+, FPN com diversos encoders | Baseline de comparacao vs SAM | [GitHub](https://github.com/qubvel-org/segmentation_models.pytorch) |
| **SAM 2** | SAM com backbone Hiera e suporte a video | Multi-scale features, arquitetura mais eficiente | [GitHub](https://github.com/facebookresearch/sam2) |
| **label-studio + SAM backend** | Anotacao interativa com SAM no navegador | Alternativa ao nosso 02_sam_interactive.py | [GitHub](https://github.com/HumanSignal/label-studio-ml-backend) |

---

## 7. Tecnicas Identificadas para Implementar

Baseado na pesquisa, estas sao as tecnicas com maior potencial de impacto no nosso pipeline:

### 7.1 Prioridade Alta

| Tecnica | Fonte | Impacto esperado | Esforco |
|---------|-------|-----------------|---------|
| **SAM-Adapter** no encoder (em vez de LoRA) | SAM-Adapter-PyTorch | Melhor adaptacao ao dominio glacial que LoRA | Medio |
| **Pre-treino com SIGSPATIAL dataset** | SIGSPATIAL Cup 2023 | Warm-start do decoder com lagos de satelite | Baixo |
| **Canais adicionais: RGB + DEM (slope, aspect, TWI)** | Literatura (Ryan 2026, Lutz 2024) | Lagos nao existem em slopes ingremes; canais seguem gradiente | Medio |
| **Prompt automatico baseado em DEM** | RSPrompter + shadow_utils.py | Pontos em depressoes topograficas = provavel lago | Medio |
| **Copy-Paste augmentation** | CALFIN, Ghiasi et al. (2021) | Multiplica dataset efetivo 2-3x | Baixo |

### 7.2 Prioridade Media

| Tecnica | Fonte | Impacto esperado | Esforco |
|---------|-------|-----------------|---------|
| **Multi-task head** (lakes + crevasses + channels juntos) | GlacierNet2 | Encoder compartilhado aprende features glaciais gerais | Alto |
| **Distance-transform loss** para crevasses/channels | CALFIN, Kervadec et al. (2019) | Melhora segmentacao de features lineares finas | Medio |
| **Anotar dados de Borebreen (Svalbard)** com nosso pipeline | Zenodo dataset | +10 imagens de drone glacial para treino | Medio |
| **U-Net baseline** para comparacao | segmentation_models.pytorch | Saber se SAM e realmente melhor que U-Net neste dominio | Medio |
| **Grounding DINO + SAM** para anotacao semi-automatica | Grounded-Segment-Anything | Anotacao mais rapida com texto: "water", "crack" | Medio |

### 7.3 Prioridade Baixa

| Tecnica | Fonte | Impacto esperado | Esforco |
|---------|-------|-----------------|---------|
| **SAM 2 Hiera-Tiny** | Wallace et al. (2025), Meta SAM2 | Backbone mais eficiente, multi-scale | Alto |
| **MobileSAM** para inferencia rapida | MobileSAM | 10x mais rapido nos 22k tiles | Baixo |
| **Label Studio + SAM** para anotacao | label-studio-ml-backend | Interface web mais completa que nosso script | Medio |

---

## 8. Conclusao da Pesquisa

### Datasets de drone glacial com anotacoes: praticamente nao existem

A maior descoberta e que **nao existem datasets publicos de drone glacial com mascaras de segmentacao** para lagos, crevasses ou canais. Isso e confirmado por:
- Wallace et al. (2025) usaram apenas 10 imagens e nao publicaram as mascaras
- Os datasets de PANGAEA/Zenodo tem ortomosaicos mas sem anotacoes
- Todos os datasets anotados sao de satelite (10-30 m), nao de drone (cm)

**Isso significa que o nosso dataset de 156 anotacoes do Schiaparelli em resolucao de 5.4 cm e unico e potencialmente publicavel.**

### O que usar para melhorar o treino

1. **SIGSPATIAL Cup 2023** — pre-treinar o decoder com lagos de satelite, depois fine-tunar com nossos dados de drone
2. **SAM-Adapter** — melhor que LoRA para domain shift grande (glacial vs natural)
3. **DEM como canal extra** — slope, aspect e TWI como informacao complementar ao RGB
4. **Prompt inteligente** — usar topografia (depressoes = lagos, linhas de fluxo = canais) em vez de grade uniforme

### Comparacao com estado da arte

| Metodo | Feicao | Resolucao | F1 | IoU |
|--------|--------|-----------|-----|------|
| **Chai et al. (2025)** SAM fine-tuned | Lagos | 10 m (Sentinel-2) | **87.8%** | — |
| **Wallace et al. (2025)** SAM 2 zero-shot | Crevasses | ~5-10 cm (drone) | — | **28%** |
| **Ryan et al. (2026)** U-Net | Lagos + canais | ~1 m (SkySat) | — | — |
| **Nosso (atual)** SAM-HQ fine-tuned | Lagos | 5.4 cm (drone) | **28.6%** | **16.7%** |
| **Nosso (atual)** SAM-HQ fine-tuned | Canais | 5.4 cm (drone) | **53.3%** | **36.4%** |
| **Nosso (atual)** SAM-HQ fine-tuned | Crevasses | 5.4 cm (drone) | **27.6%** | **16.0%** |

Nossos resultados para crevasses (IoU 0.16) estao abaixo de Wallace (IoU 0.28, zero-shot), o que sugere que ha problemas no treino alem do domain shift. Corrigir o treinamento (Fase 1 do plano) e prioritario antes de tentar tecnicas mais avancadas.

---

## Referencias e Links

### Papers
- [Wallace et al. (2025) — SAM para crevasses em drone](https://proceedings.mlr.press/v265/wallace25a.html)
- [Chai et al. (2025) — SAM para lagos supraglaciais](https://www.tandfonline.com/doi/full/10.1080/17538947.2025.2554312)
- [Muchu et al. (2024) — SAM multi-plataforma glaciologia](https://www.cambridge.org/core/journals/journal-of-glaciology/article/66D3A237ACB0975C9EE9BE19E0C2564E)
- [Ryan et al. (2026) — SkySat + deep learning meltwater](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025AV002030)
- [Lutz et al. (2024) — Lagos NE Groelandia com deep learning](https://www.mdpi.com/2072-4292/15/17/4360)

### Datasets
- [SIGSPATIAL Cup 2023 — Lagos supraglaciais](https://github.com/knowledge-computing/sigspatial-cup-2023)
- [Lutz et al. — Poligonos de lagos PANGAEA](https://doi.pangaea.de/10.1594/PANGAEA.973251)
- [NASA-IMPACT — Supraglacial segmentation](https://github.com/NASA-IMPACT/veda-ai-supraglacial_segmentation)
- [CALFIN — Calving fronts](https://github.com/daniel-cheng/CALFIN)
- [Borebreen drone dataset — Zenodo](https://zenodo.org/records/13837315)

### Repositorios GitHub
- [segment-geospatial (samgeo)](https://github.com/opengeos/segment-geospatial)
- [Geo-SAM (QGIS plugin)](https://github.com/coolzhao/Geo-SAM)
- [SAM-Adapter-PyTorch](https://github.com/tianrun-chen/SAM-Adapter-PyTorch)
- [MedSAM](https://github.com/bowang-lab/MedSAM)
- [RSPrompter](https://github.com/KyanChen/RSPrompter)
- [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [SAM 2 (Meta)](https://github.com/facebookresearch/sam2)
- [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
