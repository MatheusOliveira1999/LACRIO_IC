# Relatório Geral do Projeto LACRIO IC

**Projeto:** Mudanças na superfície de geleiras a partir de dados VANT  
**Bolsista:** Matheus Oliveira (Oceanologia - FURG)  
**Orientador:** Prof. Dr. Jorge Arigony Neto  
**Laboratório:** LaCrio - Laboratório de Monitoramento da Criosfera  
**Bolsa:** CNPq | Vigência: Set/2025 – Ago/2026  
**Data deste relatório:** 05 de Abril de 2026  

---

## 1. Objetivo do Projeto

Utilizar o modelo SAM (Segment Anything Model) para segmentar automaticamente feições supraglaciais no **Glaciar Schiaparelli** (Cordilheira Darwin, Terra do Fogo, Chile) a partir de imagens de drone (VANT), com foco em três feições:

- **Lagos supraglaciais** — acúmulo de água de degelo na superfície
- **Crevasses (fendas)** — fraturas mecânicas no gelo
- **Canais de degelo** — rede de drenagem superficial

Complementarmente, estimar a ablação superficial via análise de DEMs (Modelos Digitais de Elevação) multitemporais.

---

## 2. Dados Disponíveis

### 2.1 Mosaicos RGB (imagens de VANT)

| Ano | Arquivo | Resolução |
|-----|---------|-----------|
| 2016 | Schiaparelli_mosaic_2016.tif | 5.4 cm/pixel |
| 2017 | Schiaparelli_mosaic_2017.tif | 5.4 cm/pixel |
| 2018 | Schiaparelli_mosaic_2018.tif | 5.4 cm/pixel |
| 2019 | Schiaparelli_mosaic_2019.tif | 5.4 cm/pixel |
| 2020 | schiaparelli_mosaic_2020.tif | 5.4 cm/pixel |

### 2.2 DEMs

| Ano | Arquivo | Resolução |
|-----|---------|-----------|
| 2016–2020 | Schiaparelli_DEM_YYYY.tif | 22 cm/pixel |
| 2022 | schiaparelli_DEM_2022.tif | 22 cm/pixel |

### 2.3 Tiles gerados

- **21.798 tiles** de 512x512 pixels com overlap de 64 px
- Cobrindo os 5 anos de mosaicos (2016–2020)
- Filtro: mínimo de 70% de pixels válidos (não-NoData)

---

## 3. O Que Foi Feito

### 3.1 Pipeline implementado (6 scripts)

| Etapa | Script | Descrição | Status |
|-------|--------|-----------|--------|
| 1 | `01_create_tiles.py` | Divide mosaicos GeoTIFF em tiles 512x512 | Concluído |
| 2 | `02_sam_interactive.py` | Anotação interativa com cliques (SAM gera máscaras) | Concluído |
| 3 | `03_finetune_sam.py` | Fine-tuning do decoder do SAM | Concluído |
| 4 | `04_inference.py` | Inferência em larga escala com filtros por feição | Concluído |
| 5 | `05_reconstruct_mosaic.py` | Reconstrução dos mosaicos georreferenciados | Parcial (só lakes) |
| 6 | `06_validate.py` | Validação com métricas F1/IoU | Concluído |

Utilitários adicionais:
- `shadow_utils.py` — detecção de sombra topográfica via hillshade (DEM)
- `convert_qgis_to_masks.py` — conversão de anotações QGIS para máscaras PNG
- `config.py` — configurações centralizadas do projeto

### 3.2 Anotações manuais (ground truth)

Todas as anotações foram feitas no ano **2016** usando o script interativo (`02_sam_interactive.py`):

| Feição | Tiles anotados | Método |
|--------|---------------|--------|
| Lagos | 51 | Cliques interativos + SAM |
| Crevasses | 51 | Cliques interativos + SAM |
| Canais | 54 | Cliques interativos + SAM |
| **Total** | **156 anotações** | |

### 3.3 Modelo utilizado

- **SAM-HQ ViT-B** (Segment Anything in High Quality, Ke et al. 2023)
- Checkpoint: `sam_hq_vit_b.pth` (~375 MB)
- Estratégia de baixo consumo de VRAM (~1-2 GB):
  - Pré-computa embeddings do encoder em float16
  - Treina apenas o decoder (encoder congelado)

### 3.4 Treinamento realizado

Foram treinados 3 modelos independentes (um por feição), 30 épocas cada:

| Feição | Amostras (pos+neg) | Melhor época | Val Dice (checkpoint) |
|--------|-------------------|-------------|----------------------|
| Lakes | 51 + 51 = 102 | 2 | 0.853 |
| Crevasses | 51 + 51 = 102 | 9 | 0.495 |
| Channels | 54 + 54 = 108 | 6 | 0.673 |

### 3.5 Estudo de ablação (3 configurações)

Testadas no ano 2016 com scripts `run_ablation_stage34.sh`:

1. **HQ Only** — SAM-HQ com Dice+BCE loss, sem augmentation
2. **HQ + Augmentation** — adição de data augmentation (flips, rotações, brilho/contraste)
3. **HQ + Augmentation + LoRA** — LoRA (rank=4, alpha=16) no encoder ViT

### 3.6 Inferência e reconstrução

- Mosaicos de **lakes** foram reconstruídos para os anos 2017–2020
- Crevasses e channels não foram executados na pipeline completa de reconstrução

---

## 4. Resultados Obtidos

### 4.1 Estudo de ablação — Métricas micro (ano 2016)

| Feição | Métrica | HQ Only | HQ + Aug | HQ + Aug + LoRA |
|--------|---------|---------|----------|-----------------|
| **Lakes** | Precision | 0.106 | **0.186** | **0.186** |
| | Recall | 0.579 | **0.616** | **0.616** |
| | F1 | 0.179 | **0.286** | **0.286** |
| | IoU | 0.098 | **0.167** | **0.167** |
| **Crevasses** | Precision | 0.162 | 0.190 | **0.190** |
| | Recall | **0.605** | 0.504 | 0.504 |
| | F1 | 0.255 | **0.276** | **0.276** |
| | IoU | 0.146 | **0.160** | **0.160** |
| **Channels** | Precision | 0.295 | **0.490** | **0.490** |
| | Recall | 0.478 | **0.586** | **0.586** |
| | F1 | 0.365 | **0.533** | **0.533** |
| | IoU | 0.223 | **0.364** | **0.364** |

**Melhor configuração geral:** HQ + Augmentation (LoRA não trouxe ganho adicional).

### 4.2 Cobertura de lagos nos mosaicos reconstruídos

| Ano | Tiles usados | Pixels de lago | Cobertura (%) |
|-----|-------------|---------------|---------------|
| 2017 | 84 | 794.208 | 0.12% |
| 2018 | 46 | 663.568 | 0.11% |
| 2019 | 281 | 3.496.720 | 0.24% |
| 2020 | 102 | 1.153.584 | 0.05% |

### 4.3 Comparação com metas do projeto

| Feição | Meta F1 | Obtido F1 (melhor) | Meta IoU | Obtido IoU (melhor) | Status |
|--------|---------|-------------------|----------|--------------------|----|
| Lakes | 85–90% | **28.6%** | 75–85% | **16.7%** | Muito abaixo |
| Crevasses | 80–85% | **27.6%** | 70–80% | **16.0%** | Muito abaixo |
| Channels | 75–85% | **53.3%** | 65–75% | **36.4%** | Abaixo |

---

## 5. Diagnóstico dos Problemas

### 5.1 Colapso do decoder durante o treinamento

O problema mais grave encontrado. Analisando as curvas de treinamento:

- **Lakes:** Val Dice atingiu 0.853 na época 2, depois **colapsou para ~0.485** e ficou estagnado por 28 épocas
- **Crevasses:** Val Dice praticamente constante em ~0.477 durante 30 épocas — o modelo quase não aprendeu
- **Channels:** Mais instável, melhor resultado de 0.673 na época 6, com oscilações

O modelo está convergindo para **predições triviais** (tudo 0 ou tudo 1), o que é um sinal clássico de:
- Learning rate muito alta (1e-4 para decoder fine-tuning)
- Loss inadequada para classes extremamente desbalanceadas
- Poucas amostras de treino

### 5.2 Precision muito baixa (excesso de falsos positivos)

Em todas as feições a precision é muito baixa (10–49%), enquanto o recall é razoável (40–62%). Isso significa que o modelo está **detectando muitas regiões que não são feições reais**, incluindo:
- Sombras topográficas confundidas com lagos
- Texturas de gelo confundidas com crevasses
- Estruturas lineares diversas confundidas com canais

### 5.3 LoRA sem efeito

Os resultados de HQ+Aug e HQ+Aug+LoRA são **idênticos**, indicando que:
- Os pesos LoRA não estão sendo efetivamente atualizados, ou
- O learning rate do encoder (1e-5) é baixo demais para 30 épocas, ou
- O LoRA não está sendo injetado corretamente nas camadas do encoder

### 5.4 Dataset muito pequeno

~51 tiles anotados por feição é um volume muito baixo para fine-tuning, mesmo com augmentation. A variância alta nas métricas macro (F1_std de 0.21–0.35) confirma a instabilidade causada por poucos dados.

### 5.5 Ausência de datasets públicos compatíveis

Não existem datasets públicos de feições supraglaciais na escala de cm/pixel (VANT). Datasets de satélite (10–30 m/pixel) têm domain gap muito grande para transfer learning direto. As anotações manuais são inevitáveis.

---

## 6. Plano de Melhorias

### Fase 1 — Corrigir o treinamento (prioridade crítica)

Antes de investir em mais anotações ou pós-processamento, o treinamento precisa ser estabilizado.

| Ação | Descrição | Impacto esperado |
|------|-----------|-----------------|
| **Reduzir learning rate** | De 1e-4 para 1e-5 ou 5e-6, com warmup de 3 épocas | Evitar colapso do decoder |
| **Focal Tversky Loss** | Substituir Dice+BCE por Focal Tversky com pesos por feição (alpha/beta diferentes para lakes vs crevasses/channels) | +3–5% IoU, melhor balanço precision/recall |
| **Cosine annealing scheduler** | Decaimento suave do LR ao longo das épocas | Convergência mais estável |
| **Early stopping** | Parar treino quando val_dice não melhora por 5 épocas | Evitar overfitting e desperdício |

### Fase 2 — Aumentar e melhorar os dados

| Ação | Descrição | Impacto esperado |
|------|-----------|-----------------|
| **Mais anotações** | Anotar mais 50–100 tiles por feição, priorizando hard cases | +5–10% F1 |
| **Hard negative mining** | Usar FPs da inferência atual como negativos difíceis no próximo treino | Reduzir FPs significativamente |
| **Copy-Paste augmentation** | Recortar feições de tiles anotados e colar em tiles de fundo | Multiplica dataset efetivo por 2–3x |

### Fase 3 — Melhorar a inferência (pós-processamento)

| Ação | Descrição | Impacto esperado |
|------|-----------|-----------------|
| **Test-Time Augmentation (TTA)** | Média de predições com flips e rotação 180° | +2–4% IoU |
| **Mask refinement 2-pass** | Usar máscara do 1° pass como prompt para o 2° pass | +2–3% IoU |
| **Filtro de slope (DEM)** | Rejeitar lakes em áreas com slope > 15° | Reduzir 5–10% dos FPs em lakes |
| **CRF pós-processamento** | Refinamento de bordas usando DenseCRF | +1–3% IoU |

### Fase 4 — Análises pendentes do edital

| Ação | Descrição | Status |
|------|-----------|--------|
| **Análise de DEMs / ablação** | dH entre DEMs multitemporais, volume perdido, mapas de derretimento | Não iniciado |
| **Análise temporal** | Evolução de lagos/fendas/canais 2016–2020 | Não iniciado |
| **Correlação hidrologia-ablação** | Relação entre rede de drenagem e balanço de massa | Não iniciado |

### Fase 5 — Melhorias avançadas (se necessário)

| Ação | Descrição |
|------|-----------|
| Cross-validation 5-fold | Estimativa mais robusta das métricas |
| Pseudo-labeling / Self-training | Usar predições de alta confiança como pseudo-labels |
| Consistência multi-temporal | Cruzar detecções entre anos para filtrar FPs |
| Boundary Loss | Loss baseada em distance transform para features lineares |
| Pré-treino em datasets públicos | CALFIN, GlacierNet2 como warm-start |
| Migração para SAM2 | Backbone Hiera-Tiny com multi-scale features |

---

## 7. Cronograma Atualizado

| Fase | Atividade | Prazo estimado |
|------|-----------|---------------|
| 1 | Corrigir treinamento (LR, loss, scheduler) | 1–2 semanas |
| 2 | Expandir anotações + hard negatives | 2–3 semanas |
| 3 | Pós-processamento (TTA, slope, CRF) | 1 semana |
| 4 | Análise de DEMs e ablação | 2 semanas |
| 4 | Análise temporal integrada | 1–2 semanas |
| 5 | Documentação final e relatório IC | 1–2 semanas |

---

## 8. Resumo Executivo

O pipeline completo de segmentação com SAM-HQ foi implementado e testado para as três feições supraglaciais do Glaciar Schiaparelli. Foram criadas 156 anotações manuais e treinados 3 modelos. Os mosaicos de lagos foram reconstruídos para 4 anos (2017–2020).

**Os resultados atuais estão significativamente abaixo das metas** (F1 de 28–53% vs metas de 75–90%), com o principal gargalo sendo o colapso do decoder durante o treinamento e o excesso de falsos positivos. A configuração com SAM-HQ + Augmentation foi a melhor encontrada.

**O caminho para melhoria é claro:** estabilizar o treinamento (LR menor, Focal Tversky Loss), expandir as anotações com foco em hard cases, e aplicar pós-processamento (TTA, filtro de slope, CRF). A análise de DEMs/ablação e a análise temporal ainda não foram iniciadas e são entregas obrigatórias do edital.

---

## Referências

1. Kirillov, A. et al. (2023). "Segment Anything." ICCV 2023.
2. Ke, L. et al. (2023). "Segment Anything in High Quality." NeurIPS 2023.
3. Chai, M. et al. (2025). "Potential of SAM for supraglacial lakes." Int. J. Digital Earth.
4. Salehi, S. et al. (2017). "Tversky loss function for image segmentation using 3D FCDN."
5. Abraham, N. & Khan, N. (2019). "A novel focal Tversky loss function with improved attention U-Net."
6. Hu, E. et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models."
7. Ghiasi, G. et al. (2021). "Simple Copy-Paste is a Strong Data Augmentation Method."
