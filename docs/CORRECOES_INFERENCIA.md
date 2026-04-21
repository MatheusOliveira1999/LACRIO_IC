# Correções no Pipeline SAM - Inferência de Lagos

**Data:** 2026-02-02
**Problema:** Inferência retornando 0 feições detectadas em todos os anos (2016-2020)
**Val Dice do modelo:** 0.4124 (baixo)

---

## Diagnóstico

Três problemas críticos foram identificados no pipeline:

### 1. Mismatch treino vs. inferência (causa principal)

| Aspecto | Treino (`03_finetune_sam.py`) | Inferência (`04_inference.py`) |
|---|---|---|
| **Prompts** | 1 ponto no centroide da ground truth | 16 pontos em grade fixa (4x4) |
| **Combinação** | N/A | Média aritmética das 16 predições |
| **Resultado** | Modelo aprende a responder a pontos precisos | ~14-15 pontos caem no background, média -> ~0 |

Como a maioria dos pontos da grade não coincide com feições, a média das probabilidades fica muito abaixo do threshold de 0.5, zerando toda predição.

### 2. Falta de normalização SAM

O encoder ViT pré-treinado do SAM espera imagens normalizadas com estatísticas ImageNet:
- `pixel_mean = [123.675, 116.28, 103.53]`
- `pixel_std = [58.395, 57.12, 57.375]`

Tanto o treino quanto a inferência alimentavam o encoder com pixels brutos [0-255], gerando embeddings subótimos com o encoder congelado.

### 3. Dados insuficientes + pouca diversidade de prompts

- Apenas 51 amostras de lagos (41 treino / 10 validação)
- Apenas 1 tipo de prompt (centroide) durante treino
- Modelo parou na época 5 com Dice 0.41 (underfitting)

---

## Correções Aplicadas

### config.py

- Adicionadas constantes `PIXEL_MEAN` e `PIXEL_STD` para normalização
- Lagos `min_area`: 50 -> 20 pixels (captar poças menores)
- Lagos `max_area`: 50000 -> 100000 pixels (aceitar lagos grandes)

### 03_finetune_sam.py - Treinamento

**Normalização SAM:**
- Função `normalize_image()` aplica `(pixel - mean) / std` antes do encoder
- Embeddings pré-computados agora usam imagens normalizadas

**Diversidade de prompts (`generate_prompts()`):**

Substituiu `generate_center_prompt()` por 3 estratégias que alternam a cada batch:

| Estratégia | Descrição |
|---|---|
| `"random"` | 1-3 pontos positivos aleatórios dentro da máscara + 1 ponto negativo fora |
| `"center"` | Ponto no centroide (comportamento anterior) |
| `"bbox"` | Bounding box da máscara com perturbação aleatória |

A validação usa `"center"` para resultados determinísticos e comparáveis.

### 04_inference.py - Inferência

**Normalização SAM:**
- Mesma `normalize_image()` aplicada antes do encoder

**Nova estratégia de predição:**

| Aspecto | Antes | Depois |
|---|---|---|
| Grade de pontos | 4x4 (16 pontos, stride 256) | 8x8 (64 pontos, stride 128) |
| `multimask_output` | `False` (1 máscara) | `True` (3 máscaras, seleciona melhor por IoU) |
| Filtro IoU | Nenhum | Descarta predições com IoU < 0.5 |
| Combinação | Média aritmética | **Máximo** (basta 1 ponto detectar) |
| Threshold | 0.5 | **0.3** |

**Filtro de pós-processamento para lagos:**

| Aspecto | Antes | Depois |
|---|---|---|
| Razão azul/vermelho | > 1.3 | > 1.1 |
| Água escura | Não aceito | Aceito se brilho médio < 100 |

---

## Passos para Aplicar

Os embeddings antigos foram computados sem normalização e precisam ser refeitos:

```bash
# 1. Limpar cache de embeddings antigos
rm -rf embeddings_cache/

# 2. Re-treinar com as melhorias (recomendado 30 épocas)
python 03_finetune_sam.py --feature lakes --epochs 30

# 3. Teste rápido em 1 ano
python 04_inference.py --feature lakes --year 2016

# 4. Se resultados satisfatórios, rodar todos os anos
python 04_inference.py --feature lakes
```

---

## Resultados Obtidos

### Treinamento (lakes)

| Métrica | Antes | Depois |
|---|---|---|
| Val Dice | 0.4124 | **0.7917** |
| Melhor época | 5/20 | **11/20** |

### Inferência (lakes, 2016)

| Métrica | Antes | Depois |
|---|---|---|
| Tiles com feições | 0/2288 (0.0%) | **61/2288 (2.7%)** |
| Threshold | 0.5 | 0.3 |
| Tempo | 2h01min | 2h14min |

O aumento no tempo de inferência (~13 min) é esperado pela grade mais densa (64 vs 16 pontos) e `multimask_output=True`.

### Validação (lakes, 2016) - Problema de falsos positivos

| Métrica | Valor |
|---|---|
| F1-Score | 0.0000 |
| IoU | 0.0000 |
| Overlap GT vs predição | 1/51 tiles |

**Causa:** 61 detecções, mas quase nenhuma nos 51 tiles anotados. O modelo detectava regiões que não são lagos (sombras, gelo, rocha). Motivo: treino usava **apenas exemplos positivos** - o modelo nunca aprendeu a retornar "máscara vazia".

---

## Correções v2 - Falsos Positivos

### Problema 4: Ausência de amostras negativas no treino

O treino usava apenas tiles que contêm lagos. O modelo aprendeu: "recebeu ponto → segmenta algo", sem nunca ver a resposta correta "aqui não tem nada".

### 03_finetune_sam.py - Amostras negativas

**`collect_pairs()` com `neg_ratio=1.0`:**
- Coleta tiles sem anotação como amostras negativas (máscara toda zeros)
- Proporção 1:1 positivos/negativos (51 positivos + 51 negativos)
- Seleção aleatória com seed fixa para reprodutibilidade

**`GlacierEmbeddingDataset` atualizado:**
- Suporta `mask_path=None` → retorna máscara de zeros

**`generate_prompts()` para negativos:**
- Ponto foreground aleatório na imagem
- Target continua sendo zeros → modelo aprende a não segmentar

### 04_inference.py - Filtro espectral mais rigoroso

| Aspecto | v1 | v2 |
|---|---|---|
| Razão azul/vermelho | > 1.1 | > **1.2** |
| Água escura | brilho < 100 | brilho < **80** E azul > vermelho |
| Regiões claras | Não filtrado | **Rejeitado** se brilho > 200 |

---

## Passos para Aplicar (v2)

```bash
# 1. Limpar embeddings e modelos antigos
rm -rf embeddings_cache/

# 2. Re-treinar com amostras negativas (recomendado 30 épocas)
python 03_finetune_sam.py --feature lakes --epochs 30

# 3. Limpar predições antigas
rm -f masks/2016/lakes/tile_*.png

# 4. Re-rodar inferência
python 04_inference.py --feature lakes --year 2016

# 5. Validar
python 06_validate.py --feature lakes --year 2016
```

---

## Resultados v2

### Inferência (lakes, 2016)

| Métrica | v1 | v2 |
|---|---|---|
| Tiles com feições | 0/2288 (0.0%) | **1075/2288 (47.0%)** |
| Tempo | 2h01min | 2h15min |

### Validação (lakes, 2016)

| Métrica | v1 | v2 |
|---|---|---|
| Precision | 0.0000 | **0.1546** |
| Recall | 0.0000 | **0.2505** |
| F1-Score (micro) | 0.0000 | **0.1912** |
| IoU (micro) | 0.0000 | **0.1057** |
| F1-Score (macro) | 0.0000 | **0.0842** |

**Problemas identificados:**
1. 47% de tiles com detecção é excessivo — indica falsos positivos massivos
2. Precision 0.15 confirma: a maioria das detecções são incorretas
3. **Causa principal:** sombras topográficas estão sendo classificadas como lagos
4. Val Dice inflado (0.79) por amostras negativas que contribuem Dice=1.0 automaticamente
5. `multimask_output` inconsistente: `False` no treino, `True` na inferência

---

## Correções v3 - Máscara de Sombra DEM + Correções Estruturais

**Data:** 2026-02-04
**Problema:** Sombras topográficas confundidas com lagos (F1=0.19, Precision=0.15)

### Diagnóstico detalhado

Sombras e lagos supraglaciais compartilham assinatura espectral em RGB:
- Ambos são regiões escuras com tom azulado
- Filtro espectral (azul/vermelho > 1.2) não distingue os dois
- O modelo SAM nunca foi treinado explicitamente contra sombras

Análise da literatura confirma que sombra é o problema #1 em detecção de lagos supraglaciais:
- Stearns et al. (2023): SAM obteve F1=0.48 para lagos, com falsos positivos por sombra
- Dirscherl et al. (2021): treino reforçado em "shadow patches" para SAR
- DeepLabV3+ com filtro de sombra alcançou mIoU=94.8% (Terceiro Pólo)

### Abordagem: 3 camadas de proteção contra sombra

#### Camada 1: Máscara de sombra topográfica (DEM-based)

**Novo módulo: `shadow_utils.py`**

Usa o DEM disponível (~22 cm/pixel) para computar onde há sombra topográfica:

1. **Hillshade de Horn (1981):** Calcula iluminação para cada pixel do DEM
2. **Múltiplos ângulos solares:** 9 combinações (3 azimutes × 3 altitudes) típicas para ~54°S no verão austral

| Parâmetro | Valores | Justificativa |
|---|---|---|
| Azimute solar | 330°, 0°, 30° | Sol ao norte no hemisfério sul |
| Altitude solar | 30°, 40°, 50° | Elevação típica no verão austral |
| Threshold hillshade | < 80 | Pixel em sombra se escuro |

3. **Interseção conservadora:** Pixel é sombra somente se estiver em sombra em TODAS as 9 combinações (evita over-masking)
4. **Pré-computação por ano:** DEM aberto uma vez, máscara reutilizada para todos os tiles

**Resolução:** Máscara computada na resolução do DEM (~22cm) e redimensionada para 512×512 (resolução do tile)

#### Camada 2: Filtro de variância de textura

Sombras são texturalmente uniformes (escuro homogêneo), enquanto lagos podem ter reflexos, ondulações e variação de cor.

- Variância local computada com `cv2.blur` (janela 15×15)
- Fórmula: `Var = E[X²] - E[X]²`
- Componentes com variância média < 15.0 são removidos

#### Camada 3: Hard negatives de sombra no treino

O treino anterior usava negativos aleatórios. Agora:
- 50% dos negativos são tiles com alta cobertura de sombra (>5%)
- Tiles ordenados por cobertura de sombra (maior primeiro)
- Ensina explicitamente: "sombra ≠ lago"

### Correção estrutural: `multimask_output`

| Aspecto | Treino | Inferência v2 | Inferência v3 |
|---|---|---|---|
| `multimask_output` | `False` | `True` | **`False`** |
| Saída | 1 máscara | 3 máscaras (seleção por IoU) | 1 máscara |

O alinhamento garante que a inferência usa o mesmo modo que o treino.

### Arquivos modificados

| Arquivo | Mudança |
|---|---|
| `shadow_utils.py` | **NOVO** — Hillshade, máscara de sombra, filtro de textura |
| `config.py` | Constantes: `SHADOW_SOLAR_AZIMUTHS`, `SHADOW_SOLAR_ALTITUDES`, `SHADOW_HILLSHADE_THRESHOLD`, `SHADOW_TEXTURE_MIN_VARIANCE` |
| `04_inference.py` | Subtração de sombra + filtro textura + fix `multimask_output=False` + flags `--no-shadow`, `--shadow-threshold` |
| `03_finetune_sam.py` | `collect_pairs()` com `shadow_neg_ratio=0.5` + `_collect_shadow_negatives()` |

---

## Passos para Aplicar (v3)

```bash
# 1. Testar máscara de sombra (verifica alinhamento visual)
python shadow_utils.py
# → Abre results/shadow_mask_2016.png em QGIS sobre o mosaico

# 2. Limpar embeddings e modelos antigos
rm -rf embeddings_cache/

# 3. Re-treinar com hard negatives de sombra
python 03_finetune_sam.py --feature lakes --epochs 30

# 4. Limpar predições antigas
rm -f masks/2016/lakes/tile_*.png

# 5. Re-rodar inferência COM subtração de sombra
python 04_inference.py --feature lakes --year 2016

# 6. Validar
python 06_validate.py --feature lakes --year 2016

# Comparação A/B (SEM subtração de sombra)
python 04_inference.py --feature lakes --year 2016 --no-shadow
```
