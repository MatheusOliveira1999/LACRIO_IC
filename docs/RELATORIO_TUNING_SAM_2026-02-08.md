# Relatorio de Tuning do Pipeline SAM

**Data:** 2026-02-08  
**Projeto:** LACRIO IC - Extracao de feicoes supraglaciais  
**Escopo deste relatorio:** consolidar o que foi descoberto nos testes recentes (treino + inferencia + validacao) e definir proximos passos para melhorar resultados.

---

## 1. Contexto e objetivo

Este ciclo focou em:
- reduzir tempo de iteracao para tuning de inferencia;
- identificar se o gargalo estava em thresholds/filtros ou no modelo treinado;
- obter a melhor configuracao atual para `lakes`, `crevasses` e `channels` no ano de 2016;
- comparar resultados com as metas do projeto.

---

## 2. Alteracoes implementadas no codigo

As mudancas abaixo foram aplicadas no `04_inference.py` para acelerar e melhorar os testes:

- `--annotated-only`: roda inferencia apenas nos tiles com ground truth (modo rapido).
- remocao de mascaras antigas quando a nova predicao do tile eh vazia (evita resultado contaminado por arquivo antigo).
- correcao de performance no tensor de pontos (elimina warning de criacao lenta).
- `--pred-iou-threshold`: controla confianca minima por prompt do SAM.
- `--combine-mode {max,mean,vote}`: controla como combinar os prompts.
- filtros espectrais de lagos expostos por CLI:
- `--lakes-blue-ratio`
- `--lakes-dark-brightness`
- `--lakes-max-brightness`
- preset `--lakes-preset precision` (mais conservador).
- debug de filtro de feicao:
- `--no-feature-filter`
- filtros configuraveis de `crevasses` e `channels`:
- `--crevasses-max-brightness`
- `--crevasses-min-aspect`
- `--channels-min-aspect`

Impacto pratico:
- rodada de tuning caiu de ~136 min (2288 tiles) para ~3 min (somente tiles anotados).

---

## 3. Resultados de treino (30 epocas)

| Feicao | Amostras (pos+neg) | Melhor epoca (checkpoint) | Melhor Val Loss | Val Dice (checkpoint) |
|---|---:|---:|---:|---:|
| lakes | 51 + 51 = 102 | 7 | 0.1040 | 0.9047 |
| crevasses | 51 + 51 = 102 | 5 | 0.1373 | 0.8874 |
| channels | 54 + 54 = 108 | 6 | 0.1943 | 0.8352 |

Observacao:
- para `crevasses` e `channels`, o re-treino em 2026-02-08 foi decisivo. Antes disso, os modelos antigos (2026-02-02) estavam muito fracos em inferencia.

---

## 4. Principais descobertas por feicao (validacao 2016, modo `annotated-only`)

## 4.1 lakes

Descobertas:
- variar apenas `--threshold` (0.35 a 0.60) quase nao alterou F1 (faixa ~0.244 a ~0.247).
- ganho real veio ao ajustar tambem `pred_iou_threshold` + filtros espectrais.

Melhor configuracao testada:
- `--combine-mode max --pred-iou-threshold 0.6 --threshold 0.60 --lakes-blue-ratio 1.25 --lakes-dark-brightness 75 --lakes-max-brightness 195`

Melhor metrica:
- `Precision=0.1809 | Recall=0.4928 | F1=0.2647 | IoU=0.1525`

## 4.2 crevasses

Descobertas:
- antes do re-treino: F1 praticamente 0 em varias configuracoes.
- com `--no-feature-filter`: recall quase 1.0 e precision ~0.01, indicando excesso de falso positivo bruto.
- apos re-treino: F1 subiu para faixa ~0.26-0.29 com filtro ativo.

Melhor configuracao testada:
- `--combine-mode max --pred-iou-threshold 0.70 --threshold 0.60 --crevasses-max-brightness 130 --crevasses-min-aspect 4.5`

Melhor metrica:
- `Precision=0.2206 | Recall=0.4281 | F1=0.2912 | IoU=0.1704`

## 4.3 channels

Descobertas:
- foi a feicao com melhor resposta apos re-treino.
- aumentar conservadorismo (threshold/aspect mais altos) melhorou precision, mas caiu recall e F1 final.

Melhor configuracao testada:
- `--combine-mode max --pred-iou-threshold 0.60 --threshold 0.50 --channels-min-aspect 5.0`

Melhor metrica:
- `Precision=0.3729 | Recall=0.5706 | F1=0.4510 | IoU=0.2912`

---

## 5. Melhor configuracao atual (resumo)

| Feicao | Melhor configuracao atual (2016) | Precision | Recall | F1 | IoU |
|---|---|---:|---:|---:|---:|
| lakes | `max, iou=0.6, thr=0.60, blue_ratio=1.25, dark=75, max_brightness=195` | 0.1809 | 0.4928 | 0.2647 | 0.1525 |
| crevasses | `max, iou=0.70, thr=0.60, max_brightness=130, min_aspect=4.5` | 0.2206 | 0.4281 | 0.2912 | 0.1704 |
| channels | `max, iou=0.60, thr=0.50, min_aspect=5.0` | 0.3729 | 0.5706 | 0.4510 | 0.2912 |

---

## 6. Comparacao com metas do projeto

Metas definidas em `docs/Projeto.md`:
- lakes: F1 0.85-0.90, IoU 0.75-0.85
- crevasses: F1 0.80-0.85, IoU 0.70-0.80
- channels: F1 0.75-0.85, IoU 0.65-0.75

Estado atual (melhor encontrado):
- lakes: F1 0.2647 / IoU 0.1525
- crevasses: F1 0.2912 / IoU 0.1704
- channels: F1 0.4510 / IoU 0.2912

Conclusao:
- houve melhora clara, mas ainda ha gap grande para as metas.
- o principal gargalo atual eh precision (falsos positivos), apesar de recall razoavel em alguns cenarios.

---

## 7. O que fazer para melhorar mais (priorizado)

### Prioridade alta (curto prazo)

1. Aumentar base anotada por feicao (principalmente hard cases)
- adicionar mais positivos de baixa/media/alta dificuldade.
- adicionar negativos dificeis por feicao (sombras, textura confusa, estruturas lineares falsas).

2. Separar validacao por ano/bloco espacial
- evitar split aleatorio puro para medir generalizacao real.
- validar em blocos nao vistos no treino.

3. Calibrar thresholds por curva PR
- gerar curva precision-recall por feicao.
- escolher ponto de operacao por objetivo (max F1 ou precision minima).

4. Ajustar pos-processamento especifico
- crevasses: reforcar criterio geometrico com largura minima e continuidade local.
- channels: usar filtro de conectividade/skeleton para estruturas lineares reais.
- lakes: manter filtro espectral + revisar textura para reduzir falso positivo residual.

### Prioridade media (treino/modelo)

5. Revisar estrategia de prompts no treino para alinhar com inferencia
- incluir mais cenarios de prompts proximos da grade usada em inferencia.

6. Testar loss com foco em desbalanceamento
- comparar `Dice+BCE` com `Focal` ou `Tversky/Focal-Tversky`.

7. Data augmentation orientada a dominio
- brilho/contraste, blur leve, ruido, variacoes de cor, pequenas transformacoes geometricas.

8. Hard negative mining iterativo
- usar falsos positivos da inferencia como novos negativos no treino seguinte.

### Prioridade baixa (organizacao e confiabilidade)

9. Rastreio de experimentos
- salvar cada experimento em JSON/CSV com configuracao e metricas.

10. Relatorio automatico por rodada
- gerar tabela consolidada com melhor cfg por feicao sem sobrescrever historico.

---

## 8. Comandos recomendados (estado atual)

### 8.1 Validacao final rapida das melhores cfgs (2016)

```bash
python 04_inference.py --feature lakes --year 2016 --annotated-only \
  --combine-mode max --pred-iou-threshold 0.6 --threshold 0.60 \
  --lakes-blue-ratio 1.25 --lakes-dark-brightness 75 --lakes-max-brightness 195

python 04_inference.py --feature crevasses --year 2016 --annotated-only \
  --combine-mode max --pred-iou-threshold 0.70 --threshold 0.60 \
  --crevasses-max-brightness 130 --crevasses-min-aspect 4.5

python 04_inference.py --feature channels --year 2016 --annotated-only \
  --combine-mode max --pred-iou-threshold 0.60 --threshold 0.50 \
  --channels-min-aspect 5.0

python 06_validate.py --year 2016
cp results/validation_results.json results/validation_results_2016_tuned.json
```

### 8.2 Proximo passo para producao multianual (quando validar 2016)

```bash
# aplicar a melhor cfg por feicao para todos os anos
python 04_inference.py --feature lakes --combine-mode max --pred-iou-threshold 0.6 --threshold 0.60 --lakes-blue-ratio 1.25 --lakes-dark-brightness 75 --lakes-max-brightness 195
python 04_inference.py --feature crevasses --combine-mode max --pred-iou-threshold 0.70 --threshold 0.60 --crevasses-max-brightness 130 --crevasses-min-aspect 4.5
python 04_inference.py --feature channels --combine-mode max --pred-iou-threshold 0.60 --threshold 0.50 --channels-min-aspect 5.0
```

---

## 9. Conclusao executiva

- O pipeline evoluiu de falha total em algumas feicoes para deteccao funcional nas tres.
- `channels` esta no melhor estado atual (F1 0.4510), seguido de `crevasses` (0.2912) e `lakes` (0.2647).
- Ainda nao e suficiente para as metas finais do projeto, mas o caminho de melhoria esta claro:
- mais dados anotados de qualidade;
- hard negatives iterativos;
- ajuste orientado por curva PR e filtros por feicao;
- nova rodada de treino com foco em precision.
