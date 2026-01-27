# Progresso do Projeto - Bolsa IC LaCrio/CNPq

**Projeto:** Mudanças na superfície de geleiras a partir de dados VANT  
**Bolsista:** Matheus Oliveira  
**Orientador:** Prof. Dr. Jorge Arigony Neto  
**Vigência:** Set/2025 – Ago/2026  
**Início Implementação:** 26 de Janeiro de 2026  
**Término Previsto:** 26 de Março de 2026  
**Status Global:** 🟢 Fase 1 Concluída

---

## 📋 Alinhamento com Edital da Bolsa

| Atividade do Edital | Implementação no Projeto | Status |
|---------------------|-------------------------|--------|
| 1. Treinamento Agisoft Metashape | Dados já processados (mosaicos prontos) | ✅ |
| 2. Processamento VANT 2016-2022 | Tiles RGB 512x512 gerados | ✅ |
| 3. Estimativa ablação superficial | **Fase 7A: Análise DEMs** | 🔴 |
| 4. Mapeamento hidrologia supraglacial | **Fases 2-6: ML com SAM** | 🔴 |

---

## 📊 Resumo de Progresso

| Etapa | Status | Início | Fim | Horas |
|-------|--------|--------|-----|-------|
| Fase 1 - Setup e Preparação | 🟢 Concluído | 26/01 | 26/01 | 10h |
| Fase 2 - Anotação Interativa | 🔴 Não Iniciado | - | - | 8h |
| Fase 3 - Fine-tuning SAM | 🔴 Não Iniciado | - | - | 5h |
| Fase 4 - Inferência ML | 🔴 Não Iniciado | - | - | 10h |
| Fase 5 - Reconstrução | 🔴 Não Iniciado | - | - | 8h |
| Fase 6 - Validação | 🔴 Não Iniciado | - | - | 6h |
| **Fase 7A - Análise DEMs (NOVO)** | 🔴 Não Iniciado | - | - | 10h |
| Fase 7B - Análise Temporal | 🔴 Não Iniciado | - | - | 8h |
| Fase 8 - Documentação | 🔴 Não Iniciado | - | - | 10h |

**Progresso Geral:** █░░░░░░░░░ 12%

---

## 📅 Cronograma Detalhado

### Semana 1 (26/01 - 01/02): Setup e Preparação ✅
> **Objetivo:** Configurar ambiente e preparar dados

| Tarefa | Status | Descrição |
|--------|--------|-----------|
| [x] 1.1 Ambiente Conda | 🟢 | `sam_glaciar` criado |
| [x] 1.2 PyTorch + CUDA | 🟢 | Ambiente configurado |
| [x] 1.3 Modelo SAM | 🟢 | `sam_vit_b_01ec64.pth` baixado |
| [x] 1.4 `config.py` | 🟢 | Configurações do projeto |
| [x] 1.5 `01_create_tiles.py` | 🟢 | Script de tiling |
| [x] 1.6 Tiles gerados | 🟢 | **21.798 tiles** (2016-2020) |

**Entregável:** ✅ Tiles em `/tiles/YYYY/`

---

### Semana 2 (02/02 - 08/02): Anotação Interativa
> **Objetivo:** Criar ground truth para ML (Atividade 4 do edital)

| Tarefa | Status | Descrição |
|--------|--------|-----------|
| [ ] 2.1 `02_sam_interactive.py` | 🔴 | Interface de anotação |
| [ ] 2.2 Anotar lagos (20+ tiles) | 🔴 | Feições azuis/turquesa |
| [ ] 2.3 Anotar fendas (15+ tiles) | 🔴 | Feições lineares escuras |
| [ ] 2.4 Anotar canais (15+ tiles) | 🔴 | Rede de drenagem |
| [ ] 2.5 Revisar qualidade | 🔴 | Verificar anotações |

**Entregável:** 50+ máscaras em `/masks/YYYY/annotations/`

---

### Semana 3 (09/02 - 15/02): Fine-tuning SAM
> **Objetivo:** Adaptar SAM para dados do Schiaparelli

| Tarefa | Status | Descrição |
|--------|--------|-----------|
| [ ] 3.1 `03_finetune_sam.py` | 🔴 | Script de treinamento |
| [ ] 3.2 GlacierDataset | 🔴 | Carregar pares tile-máscara |
| [ ] 3.3 Treinar lagos | 🔴 | 20 épocas, lr=1e-4 |
| [ ] 3.4 Treinar fendas | 🔴 | Ajustar hiperparâmetros |
| [ ] 3.5 Treinar canais | 🔴 | Validar cada feição |

**Entregável:** Modelo em `/models/sam_finetuned_best.pth`

---

### Semana 4 (16/02 - 22/02): Inferência em Larga Escala
> **Objetivo:** Aplicar SAM a todos os tiles

| Tarefa | Status | Descrição |
|--------|--------|-----------|
| [ ] 4.1 `04_inference.py` | 🔴 | Script de inferência |
| [ ] 4.2-4.6 Processar 2016-2020 | 🔴 | Lagos, fendas, canais |

**Entregável:** Máscaras em `/masks/YYYY/{feature}/`

---

### Semana 5 (23/02 - 01/03): Reconstrução
> **Objetivo:** Unir tiles e criar mosaicos de feições

| Tarefa | Status | Descrição |
|--------|--------|-----------|
| [ ] 5.1 `05_reconstruct_mosaic.py` | 🔴 | Reconstrução |
| [ ] 5.2-5.4 Reconstruir feições | 🔴 | GeoTIFF com CRS |
| [ ] 5.5 Pós-processamento | 🔴 | Morfologia, limpeza |

**Entregável:** Mosaicos em `/results/YYYY/`

---

### Semana 6 (02/03 - 05/03): Validação
> **Objetivo:** Avaliar qualidade das segmentações

| Tarefa | Status | Descrição |
|--------|--------|-----------|
| [ ] 6.1 F1-score lagos | 🔴 | Meta: 85-90% |
| [ ] 6.2 F1-score fendas | 🔴 | Meta: 80-85% |
| [ ] 6.3 F1-score canais | 🔴 | Meta: 75-85% |
| [ ] 6.4 Ajustar thresholds | 🔴 | Otimizar parâmetros |

**Entregável:** Relatório de métricas F1/IoU

---

### 🆕 Semana 6-7 (05/03 - 12/03): Análise de DEMs e Ablação
> **Objetivo:** Estimativa da ablação superficial (Atividade 3 do edital)

| Tarefa | Status | Descrição |
|--------|--------|-----------|
| [ ] 7A.1 `06_dem_analysis.py` | 🔴 | Script de análise de DEMs |
| [ ] 7A.2 Diferença de elevação | 🔴 | dH = DEM(t2) - DEM(t1) |
| [ ] 7A.3 Mapa de derretimento | 🔴 | Variação espacial ablação |
| [ ] 7A.4 Calcular volume perdido | 🔴 | Integração espacial de dH |
| [ ] 7A.5 Correlação hidrologia | 🔴 | Ablação vs rede de drenagem |
| [ ] 7A.6 Série temporal balanço | 🔴 | Gráficos 2016-2022 |

**DEMs disponíveis:**
- `Schiaparelli_DEM_2016.tif` (165 MB)
- `Schiaparelli_DEM_2017.tif` (155 MB)
- `Schiaparelli_DEM_2018.tif` (150 MB)
- `Schiaparelli_DEM_2019.tif` (347 MB)
- `schiaparelli_DEM_2020.tif` (400 MB)
- `schiaparelli_DEM_2022.tif` (350 MB)

**Entregável:** Mapas de ablação e séries temporais de balanço de massa

---

### Semana 7-8 (12/03 - 18/03): Análise Temporal Integrada
> **Objetivo:** Comparar evolução 2016-2022 (Atividade 4 do edital)

| Tarefa | Status | Descrição |
|--------|--------|-----------|
| [ ] 7B.1 Área lagos/ano | 🔴 | Série temporal |
| [ ] 7B.2 Extensão fendas/ano | 🔴 | Evolução geométrica |
| [ ] 7B.3 Rede de drenagem | 🔴 | Conectividade canais |
| [ ] 7B.4 Correlação ablação-hidrologia | 🔴 | Balanço massa vs drenagem |
| [ ] 7B.5 Gráficos temporais | 🔴 | Matplotlib/Seaborn |
| [ ] 7B.6 Tendências | 🔴 | Análise estatística |

**Entregável:** Séries temporais integradas

---

### Semana 8 (18/03 - 26/03): Documentação Final
> **Objetivo:** Consolidar resultados e documentar

| Tarefa | Status | Descrição |
|--------|--------|-----------|
| [ ] 8.1 Metodologia | 🔴 | Descrição técnica |
| [ ] 8.2 README | 🔴 | Instruções de uso |
| [ ] 8.3 Figuras finais | 🔴 | Mapas e visualizações |
| [ ] 8.4 Relatório IC | 🔴 | Relatório final bolsa |
| [ ] 8.5 Apresentação | 🔴 | Slides MPU/FURG |

**Entregável:** Relatório final da bolsa IC

---

## 📁 Estrutura de Arquivos

```
LACRIO IC/
├── config.py                    # ✅ Configurações
├── 01_create_tiles.py           # ✅ Tiling
├── 02_sam_interactive.py        # Anotação
├── 03_finetune_sam.py           # Fine-tuning
├── 04_inference.py              # Inferência
├── 05_reconstruct_mosaic.py     # Reconstrução
├── 06_dem_analysis.py           # 🆕 Análise DEMs/Ablação
├── mosaicos_DEMs_Schiaparelli/  # Dados fonte (.tif)
├── tiles/                       # ✅ Tiles RGB
├── masks/                       # Máscaras ML
├── models/                      # Modelos treinados
└── results/                     # Resultados finais
```

---

## 🎯 Marcos do Projeto

| Marco | Data Prevista | Status |
|-------|---------------|--------|
| M1: Ambiente configurado | 01/02/2026 | 🟢 26/01 |
| M2: Anotações completas | 08/02/2026 | 🔴 |
| M3: Modelo treinado | 15/02/2026 | 🔴 |
| M4: Inferência completa | 22/02/2026 | 🔴 |
| M5: Mosaicos reconstruídos | 01/03/2026 | 🔴 |
| M6: Métricas validadas | 05/03/2026 | 🔴 |
| **M7: Análise DEMs/Ablação** | 12/03/2026 | 🔴 |
| M8: Análise temporal | 18/03/2026 | 🔴 |
| M9: Projeto entregue | 26/03/2026 | 🔴 |

---

## 📝 Notas e Observações

### Atualizações Recentes
- **26/01/2026:** Fase 1 concluída - 21.798 tiles gerados
- **27/01/2026:** Projeto alinhado com edital da bolsa IC
- **27/01/2026:** Adicionada Fase 7A (Análise DEMs/Ablação)

### Decisões Técnicas
- Modelo SAM: `vit_b` (dev) / `vit_h` (prod)
- Tile size: 512x512, overlap 64px
- DEMs: 6 anos disponíveis (2016-2022)

### Riscos Identificados
1. Volume de dados (~25 GB tiles)
2. Tempo de inferência GPU
3. Alinhamento DEMs multitemporais

---

*Última atualização: 27/01/2026*
