# Diagnostico: Discrepancia entre Treino (Dice 0.87) e Inferencia (F1 0.12)

## Resumo do Problema

O modelo SAM-HQ fine-tuned atinge **Val Dice 0.87** durante o treino, mas na validacao real contra ground truth a **F1 cai para 0.12**. Este documento analisa as causas raiz.

---

## Causa Raiz Principal: SAM e um modelo interativo, nao semantico

O SAM (Segment Anything Model) foi projetado como um modelo **interativo** - ele precisa de um **prompt** (ponto, bbox, mascara) que diga ONDE esta o objeto. Ele nao faz deteccao automatica.

### Como funciona no treino:
- O prompt e gerado a partir do **ground truth** (bbox ao redor do lago, ponto no centroide)
- O modelo so precisa **refinar bordas** - ele ja sabe onde o lago esta
- Val Dice 0.87 e medido com prompt no **centro exato** do lago

### Como funciona na inferencia:
- **Nao temos ground truth** - nao sabemos onde estao os lagos
- Tentamos grade de 64 pontos cegos, detector espectral, etc.
- Nenhuma abordagem consegue replicar a qualidade dos prompts do treino

**Conclusao**: O Dice 0.87 mede a capacidade do SAM de *refinar bordas dado um prompt correto*, nao de *encontrar lagos automaticamente*.

---

## Diagnostico por Tile (51 tiles com ground truth)

| Categoria | Qtd | % | Descricao |
|-----------|-----|---|-----------|
| Bons (overlap razoavel) | 20 | 39% | Predicao acerta local, mas tamanho varia |
| Zero overlap | 13 | 25% | Predicao existe mas em local errado |
| Oversegmentacao (>5x GT) | 13 | 25% | Acerta local mas segmenta area 8-168x maior |
| Sem predicao | 5 | 10% | Nenhum candidato encontrado |

### Dados agregados:
- **Total GT**: 48,687 pixels
- **Total Pred**: 247,395 pixels
- **Ratio Pred/GT**: 5.1x (prediz 5x mais area que o real)

---

## Tres Problemas Especificos

### 1. Oversegmentacao (13 tiles, 25%)

Exemplos:
- `tile_001905`: GT=160px, Pred=26,862px (**168x**). Recall=0.95, Precision=0.01
- `tile_001612`: GT=267px, Pred=17,677px (**66x**). Recall=0.94, Precision=0.01
- `tile_001727`: GT=920px, Pred=25,964px (**28x**). Recall=0.96, Precision=0.03

**O que acontece**: O detector espectral encontra uma regiao azulada, gera um bbox, e o SAM segmenta uma area muito maior que o candidato original. O SAM "expande" a predicao para alem da regiao azul, segmentando gelo/neve adjacente.

**Causa**: O SAM foi treinado com bboxes que envolvem o lago inteiro. Quando recebe um bbox pequeno, ele interpreta como "ha um objeto aqui, segmente-o por completo" e expande alem do bbox.

### 2. Zero overlap (13 tiles, 25%)

Exemplos:
- `tile_000570`: GT=916px, Pred=3,256px, Inter=0px
- `tile_002071`: GT=1,146px, Pred=7,648px, Inter=0px
- `tile_001076`: GT=1,681px, Pred=1,104px, Inter=0px

**O que acontece**: O detector espectral encontra uma regiao azulada que NAO e um lago (sombra, gelo azulado, artefato), o SAM refina essa regiao, e a predicao cai em local completamente diferente do GT.

**Causa**: O detector espectral tem falsos positivos - nao toda regiao azul e um lago.

### 3. Sem predicao (5 tiles, 10%)

Tiles: 000037, 000560, 000671, 001508, 001624

**O que acontece**: O detector espectral nao encontra nenhum candidato (min_area=100, B/R>1.4), mesmo havendo lago no GT.

**Causa**: Alguns lagos sao muito pequenos (<100px) ou tem assinatura espectral fraca.

---

## Abordagens Testadas

| Abordagem | P | R | F1 | Problema |
|-----------|---|---|----|----|
| Grade 8x8 (default) | 0.14 | 0.25 | 0.18 | Falsos positivos em todo tile |
| Grade 8x8 + mean + thr 0.45 | - | - | 0.00 | Sinal diluido, nada detectado |
| Precision preset (vote+0.85) | - | - | ~0.00 | IoU threshold mata tudo |
| Propose-refine (B/R>1.15) | 0.07 | 0.40 | 0.12 | Spectral muito permissivo |
| Propose-refine (B/R>1.4) + clip | 0.08 | 0.37 | 0.13 | Overseg + zero overlap |
| Spectral sozinho | 0.03 | - | 0.03 | Sem refinamento |
| SAM noprompt | - | - | 0.01 | SAM precisa de prompt |
| SAM centro | - | - | 0.05 | Prompt aleatorio, nao no lago |

---

## Por que o Treino Parece Bom mas a Inferencia Falha

```
TREINO:                          INFERENCIA:
                                 
[GT Mask] --> [Gera Prompt] -->  [???] --> [Gera Prompt] ???
                |                              |
            Bbox/Centro                  Nao tem GT!
            NO LAGO                      Tenta espectral/grid
                |                              |
            [SAM Decoder]               [SAM Decoder]  
                |                              |
           Dice = 0.87                    F1 = 0.12
```

O gap entre treino e inferencia existe porque o **prompt de qualidade** (baseado no GT) nao esta disponivel na inferencia.

---

## Caminhos para Solucao

### Opcao A: Treinar SAM como segmentador semantico
- Modificar o treino para usar grade de pontos (como na inferencia)
- Amostras negativas: tiles SEM lago devem gerar mascara vazia
- O modelo aprende QUANDO segmentar (nao so COMO)
- **Impacto**: Alto, mas requer re-treino completo

### Opcao B: Usar U-Net/DeepLabv3 em vez de SAM
- Modelos de segmentacao semantica nao precisam de prompts
- Encoder pre-treinado (ResNet/EfficientNet) + decoder leve
- Mais adequado para "encontrar todos os lagos automaticamente"
- **Impacto**: Mudanca de arquitetura, mas resolve o problema raiz

### Opcao C: Melhorar o detector de candidatos
- Usar features mais robustas que B/R (ex: NDWI, textura, forma)
- Treinar um classificador simples (RF/SVM) para filtrar candidatos
- SAM so refina candidatos de alta confianca
- **Impacto**: Medio, mantem SAM mas melhora a deteccao

### Opcao D: Two-stage com classificador de tiles
- Stage 1: Classificador binario por tile (tem lago / nao tem)
- Stage 2: SAM com bbox nos tiles positivos
- **Impacto**: Reduz falsos positivos drasticamente

---

## Recomendacao

**Opcao A** e a mais alinhada com o projeto atual. Modificar o treino para:
1. Usar grade densa de pontos como prompts (nao so bbox/centro)
2. Incluir tiles negativos (sem feicao) para ensinar o modelo a dizer "nada aqui"
3. Loss penaliza falsos positivos em tiles negativos

Isso faz o modelo aprender a diferenca entre "lago" e "nao-lago" em vez de so refinar bordas.
