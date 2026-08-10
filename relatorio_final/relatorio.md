---
title: ""
lang: pt-BR
---

# Relatório Final de Iniciação Científica

**Mudanças na superfície de geleiras a partir de dados VANT: segmentação automática de fendas supraglaciais e sua relação com a ablação no Glaciar Schiaparelli (Cordilheira Darwin, Chile)**

| | |
|---|---|
| **Bolsista** | Matheus Oliveira |
| **Orientador** | Prof. Dr. Jorge Arigony Neto |
| **Laboratório** | LaCrio — Laboratório de Monitoramento da Criosfera, FURG |
| **Vigência da bolsa** | Setembro/2025 – Agosto/2026 (CNPq) |
| **Data do relatório** | Agosto de 2026 |

## Resumo

Este projeto desenvolveu um pipeline completo de aprendizado profundo para mapear fendas (crevasses) supraglaciais em ortomosaicos de VANT de altíssima resolução (8 cm/pixel) do Glaciar Schiaparelli, e usou os produtos resultantes para quantificar, pela primeira vez neste glaciar, a relação entre densidade de fendas e afinamento da superfície. Uma U-Net de seis canais (RGB + relevo, declividade e curvatura derivados do MDE), treinada exclusivamente com dados de 2016 e avaliada sob validação cruzada **espacial** com zona de exclusão anti-vazamento, atingiu F1 = 0,654 ± 0,014 (precisão 0,613; revocação 0,702; AUC-PR 0,640; AUC-ROC 0,928), com transferência temporal estável para 2017–2019 (F1 0,66–0,72) sem re-treinamento. Uma auditoria sistemática dos rótulos revelou que 30% dos "negativos" de treinamento continham fendas reais; sua correção elevou a precisão medida em 5 pontos percentuais. Sobre os produtos cartográficos de quatro anos (2016–2019), as análises glaciológicas mostram: (i) **acoplamento episódico entre fendilhamento e afinamento** — no ano de afinamento máximo (2016→17), blocos de 100 m mais fendilhados afinaram ~2 m a mais que gelo sem fendas (ρ de Spearman = −0,58; ρ parcial controlando elevação = −0,39; p < 10⁻⁸), com o acoplamento ausente nos anos de afinamento moderado; (ii) **fragmentação progressiva da rede de fendas** (+40% em número, −16% em área total, mediana de área de 1,95 para 0,65 m²); e (iii) **expansão do campo de fendas para montante**, com as fendas novas de 2019 surgindo ~18 m acima e ~250 m mais distantes do terminus que as persistentes. Um experimento de consistência intra-anotador (re-anotação às cegas) resultou em auto-concordância de F1 = 0,58 (estrita) e 0,79 (tolerância de 4 px) — o modelo, portanto, opera no teto de ruído do próprio ground truth. O conjunto de dados anotados constitui o primeiro benchmark de segmentação de fendas em VANT com validação espacial e transferência temporal documentadas.

**Palavras-chave:** fendas supraglaciais; VANT; aprendizado profundo; U-Net; validação cruzada espacial; ablação; Cordilheira Darwin.

\newpage

# 1. Introdução e objetivos

Fendas supraglaciais são fraturas mecânicas do gelo que registram o campo de tensões do fluxo glacial e participam ativamente dos processos de ablação: aumentam a rugosidade e a área exposta da superfície, canalizam água de degelo e aceleram a transferência de calor (Colgan et al., 2016). A quantificação direta da relação entre densidade de fendas e perda de elevação da superfície, no entanto, permanece pouco explorada — em parte pela dificuldade de mapear fendas de forma exaustiva e reprodutível.

Ortomosaicos de VANT com resolução centimétrica tornam esse mapeamento possível. Até onde foi possível verificar por varredura sistemática de repositórios de dados e da literatura (agosto de 2026), **não existe nenhum benchmark público de segmentação de fendas em imagens de VANT** — isto é, nenhum conjunto de dados publicamente disponível que pareie imagens de drone de glaciares com máscaras de fendas anotadas^[Casos-limite verificados em 09/08/2026: (i) Wallace et al. (2025) anotaram máscaras em 10 imagens UAV (Borebreen, Svalbard), mas o repositório oficial contém apenas código e o artigo não possui declaração de disponibilidade de dados — as máscaras permanecem privadas; (ii) imagens brutas de drone de glaciares fendilhados já são públicas *sem anotações* (Hann et al., 2024; DataverseNO, CC0) — o inédito são as máscaras, não as imagens; (iii) para sensor aéreo *tripulado* existem mapas públicos de fendas a 0,2 m nos Alpes (Baumhoer et al., 2025; DLR GeoService, CC-BY), porém são predições de modelo, sem ground truth pareado; (iv) produtos públicos de fendas em *satélite* existem (ex.: Zenodo 7254130, 6779087); (v) Duan et al. (2025): a declaração de disponibilidade do próprio artigo informa "Data will be made available on request" (verificado no texto em 09/08/2026) — dados não públicos. Repositórios varridos: Zenodo, DataverseNO/Harvard Dataverse, Figshare, Mendeley Data, IEEE DataPort, PANGAEA, Papers With Code, Roboflow, HuggingFace, Kaggle, GitHub, arXiv, OpenAlex, DOAJ (buscas em inglês e chinês).]. As únicas anotações desse tipo relatadas na literatura (Wallace et al., 2025: 10 imagens do Glaciar Borebreen, Svalbard) não foram disponibilizadas publicamente; esse único resultado publicado com métricas em dados de drone usa o modelo de fundação SAM 2 em modo zero-shot, com desempenho limitado (DSC 0,43), e seus autores apontam explicitamente a necessidade de modelos supervisionados dedicados.

Este projeto, vinculado ao edital da bolsa (atividades: 1 — processamento Agisoft Metashape; 2 — processamento dos levantamentos VANT 2016–2022; 3 — estimativa de ablação superficial; 4 — mapeamento da hidrologia supraglacial por aprendizado de máquina), teve como objetivos específicos:

1. Desenvolver e validar um modelo de segmentação semântica de fendas supraglaciais em ortomosaicos VANT de 8 cm do Glaciar Schiaparelli;
2. Gerar produtos cartográficos multi-anuais (2016–2019) de fendas com protocolo de avaliação rigoroso;
3. Estimar a ablação superficial por diferenciação de MDEs e **quantificar a relação densidade de fendas × afinamento** (atividade 3);
4. Caracterizar a evolução temporal da rede de fendas (fragmentação, migração, orientação) como componente da análise integrada (atividade 4).

# 2. Dados e métodos

## 2.1 Área de estudo e dados

O Glaciar Schiaparelli (~54,4°S, 70,9°W, Cordilheira Darwin, Terra do Fogo, Chile) foi levantado por VANT em campanhas anuais (2016–2020), processadas em Agisoft Metashape (atividade 1 do edital). Os produtos por ano são: ortomosaico RGB (resolução nativa 5,4–8,2 cm/pixel) e MDE (~31 cm/pixel), em EPSG:32719. Para comparabilidade interanual, os ortomosaicos 2016–2019 foram reamostrados para grade comum de **8,0 cm/pixel** e recortados em ~2 200 tiles de 512×512 pixels por ano, com sobreposição de 64 px (atividade 2). A área coberta é de ~4,3 km² (Figura 1).

![](../results/figures/relatorio/mapa_area_estudo_qgis.png){width=16cm}

*Figura 1 — Localização e área de estudo: Glaciar Schiaparelli, Cordilheira Darwin (Terra do Fogo, Chile; ~54,40°S, 70,88°W). Frame principal: ortomosaico VANT de 2016 (8 cm/pixel) com contornos de elevação de 50 m e limite do glaciar; grade de coordenadas UTM 19S (EPSG:32719). Insets: contexto continental e regional (mapa-base © OpenStreetMap/CARTO).*

## 2.2 Anotação e auditoria de rótulos

As fendas de 2016 foram anotadas manualmente pixel a pixel com ferramenta própria (`annotate.py`), sob guia estrito de definição (feição linear, famílias paralelas, sombra profunda; exclusão de sastrugi, canais de drenagem e sombras de morena). O conjunto final de treinamento/validação contém **218 tiles de 2016 (141 positivos + 77 negativos explícitos)**.

Dois problemas de qualidade de rótulo foram identificados e corrigidos durante o projeto, e são reportados como parte do método:

- **Auditoria dos negativos implícitos.** O amostrador de treinamento completava o conjunto com tiles aleatórios não verificados assumidos como negativos (30% do dataset). Uma auditoria visual sistemática (painel duplo-cego com critério do guia, seguida de revisão humana de 100% dos 65 tiles) revelou que **20,4% dos "falsos positivos" medidos eram fendas reais não rotuladas**. A correção (32 tiles re-anotados como positivos; 33 confirmados negativos) elevou a precisão da validação cruzada em +5,0 p.p. e o pool de negativos foi congelado (apenas negativos verificados).
- **Circularidade no GT interanual.** As anotações de referência de 2017–2019 (38/39/40 tiles), revisadas em 2026 a partir das predições de um modelo anterior, mostraram-se parcialmente idênticas a essas predições (~40% dos tiles com máscara pixel-idêntica). As métricas interanuais são portanto reportadas **com esse viés declarado**, e valores anteriores inflados (F1 0,83–0,87) foram invalidados.

## 2.3 Modelo e treinamento

O modelo é uma **U-Net com encoder ResNet-34** (pesos ImageNet) e **seis canais de entrada**: RGB + relevo local, declividade e curvatura derivados do MDE (normalizados por z-score; canais extras inicializados com a média dos pesos RGB). Treinamento: perda BCE + Tversky (α_FP = 0,7; α_FN = 0,3), taxas de aprendizado discriminativas (encoder 10⁻⁵, decoder 10⁻⁴), augmentação geométrica (RGB+MDE) e cromática calibrada (somente RGB, incluindo gradiente direcional de brilho para iluminação heterogênea de voo), 200 épocas. O modelo final (época 107) treinou em ~1 h em uma GPU GTX 1650.

## 2.4 Protocolo de avaliação

A avaliação principal é uma **validação cruzada espacial 5-fold com zona de exclusão** (Roberts et al., 2017): os tiles são agrupados em blocos espaciais de 2 048 px e todo tile de treinamento cujo enquadramento sobrepõe qualquer tile de validação é removido do fold (Fig. 2). Esse desenho elimina o vazamento por autocorrelação espacial que infla métricas sob particionamento aleatório — efeito medido em +14,6 a +27,9% de F1 para CNNs em ortoimagens de VANT (Kattenborn et al., 2022). Complementarmente: transferência temporal (modelo de 2016 aplicado a 2017–2019 sem re-treinamento) e métricas com tolerância espacial de 4 px (~32 cm), seguindo a convenção de estruturas lineares finas (clIoU@4px do benchmark OmniCrack30k; "relaxed precision/recall" com ρ = 3 px na literatura de extração de estradas desde Wiedemann et al., 1998).

Na inferência de produção usa-se **TTA** (média de 4 orientações; +0,9 p.p. em precisão, revocação e F1), **limiar 0,6** (centro do platô da curva OOF; Fig. 3) e **reconstrução do mosaico por média de probabilidades ponderada por janela de Hann** — que substitui o mosaico por máximo binário, cuja precisão nas zonas de sobreposição de tiles (25,8% da área) cai para 0,39 contra 0,72 nas zonas de consenso; o blend recupera +1,7 p.p. de precisão no produto final (−8,9% de pixels de falso positivo).

## 2.5 Análises glaciológicas

A ablação superficial (atividade 3) foi estimada por diferenciação de MDEs consecutivos reamostrados para grade comum de 2 m. O viés vertical de co-registro, medido em blocos de terreno estável, é ≤ |0,04| m em todos os pares — duas ordens de grandeza abaixo do sinal. A **correlação densidade de fendas × dH** foi computada em blocos de 100 m dentro do domínio de gelo, definido por proxy independente (afinamento < −0,5 m em intervalos de anos complementares ao par analisado, evitando circularidade), com correlação de Spearman bruta e **parcial controlando a elevação**, e contraste gelo-sem-fenda vs. quartil superior de densidade (Mann-Whitney). Script reprodutível: `11_density_dh_analysis.py`. As demais análises (inventário, morfometria, orientação axial, persistência, hipsometria) derivam das máscaras finais de 8 cm e dos MDEs.

\newpage

# 3. Resultados: segmentação

## 3.1 Validação cruzada espacial

![](../results/figures/relatorio/mapa_folds_cv_qgis.png){width=16cm}

*Figura 2 — Área de estudo e desenho da validação cruzada espacial: blocos de 2 048 px atribuídos aos 5 folds sobre o ortomosaico de 2016, com recortes das morfologias de fenda cobertas por cada fold (molduras na cor do fold). Coordenadas UTM 19S (EPSG:32719).*

**Tabela 1 — Métricas da validação cruzada espacial 5-fold (2016, dataset auditado).**

| Métrica | Fórmula | Valor (média ± dp dos folds) |
|---|---|---|
| Precisão | TP/(TP+FP) | 0,613 ± 0,023 |
| Revocação | TP/(TP+FN) | 0,702 ± 0,032 |
| F1 | 2PR/(P+R) | **0,654 ± 0,014** |
| AUC-PR (OOF) | — | 0,640 |
| AUC-ROC (OOF) | — | 0,928 |
| Precisão com tolerância 4 px (~32 cm) | — | ~0,90 |

**Tabela 2 — Desempenho por fold espacial e transferência temporal.**

| Partição | n tiles val. | Precisão | Revocação | F1 |
|---|---|---|---|---|
| Fold 1 | 44 | 0,642 | 0,687 | 0,663 |
| Fold 2 | 43 | 0,617 | 0,689 | 0,651 |
| Fold 3 | 44 | 0,594 | 0,665 | 0,628 |
| Fold 4 | 44 | 0,580 | 0,760 | 0,658 |
| Fold 5 | 43 | 0,633 | 0,710 | 0,669 |
| 2016 (in-sample, referência)¹ | 218 | 0,678 | 0,768 | 0,720 |
| 2017 (transf.)² | 38 | 0,674 | 0,684 | 0,679 |
| 2018 (transf.)² | 39 | 0,669 | 0,751 | 0,708 |
| 2019 (transf.)² | 40 | 0,672 | 0,656 | 0,664 |

¹ *Medido nos tiles de treinamento; reportado apenas para comparabilidade com a literatura in-domain.* ² *GT interanual com viés declarado (Seção 2.2); inferência com TTA e limiar 0,6.*

![](../results/figures/relatorio/fig02_curvas_prf1_threshold.png){width=11cm}

*Figura 3 — Precisão, revocação e F1 em função do limiar de binarização (curvas out-of-fold; banda = ±1 dp entre folds). O platô de F1 entre 0,45 e 0,70 permite escolher o ponto operacional pela precisão; o limiar de produção (0,6) está marcado.*

O F1 espacial de 0,654 ± 0,014 é estável entre folds (0,628–0,669) e a transferência temporal não degrada (0,66–0,72 em 2017–2019, com precisão uniformizada em ~0,67) — evidência de que o modelo de um único ano, com canais topográficos e augmentação cromática dirigida, generaliza entre condições de superfície muito distintas (neve fresca em 2017; gelo exposto em 2018/2019).

## 3.2 Análise qualitativa e modos de erro

![](../results/figures/relatorio/fig03_qualitativa.png){width=15cm}

*Figura 4 — Avaliação qualitativa em cenas representativas (ortofoto | referência manual | predição | sobreposição): (a) fendas largas com detritos, F1 = 0,91; (b) fraturas finas em gelo limpo, F1 = 0,90; (c) campo denso de fraturas, F1 = 0,70; (d) fendas sob neve — falso negativo difícil, R = 0,33; (e) seracs e sombra, R = 0,35. Convenção de cores fixa: referência em turquesa, predição em azul.*

![](../results/figures/relatorio/fig04_falsos_positivos.png){width=15cm}

*Figura 5 — Modos típicos de falso positivo: sombras de relevo, morenas/detritos escuros, canais supraglaciais, textura de neve. 68,6% dos pixels de falso positivo situam-se a ≤ 4 px (~32 cm) da referência — o erro dominante é ambiguidade de borda, não detecção espúria.*

## 3.3 Posicionamento na literatura

**Tabela 3 — Comparação com a literatura de segmentação de fendas (sem ordenação: sensores, métricas e protocolos diferem).**

| Estudo | Sensor / resolução | Métrica reportada | Protocolo |
|---|---|---|---|
| **Este trabalho (2026)** | VANT RGB+MDE, 8 cm | **F1 0,654 ± 0,014**; AUC-ROC 0,928; P@4px ≈ 0,90 | **CV espacial c/ exclusão + transferência 3 anos; GT auditado** |
| Wallace et al. 2025 | VANT RGB, 10 cm | DSC 0,43 / IoU 0,28 | SAM 2 zero-shot; n = 10 imagens |
| Duan et al. 2025 (C-TransUNet) | VANT RGB 3 cm (LiDAR apenas p/ profundidade 3D pós-segmentação) | F1 0,871 (U-Net clássica no mesmo teste: 0,765) | In-domain, 2 ROIs de um único terminus; partição por patch (janela deslizante 512 px, 7:2:1) — vazamento espacial provável; transferência a outros glaciares apenas qualitativa, sem GT |
| Baumhoer et al. 2025 | Aéreo óptico, 20 cm | Acurácia balanceada 0,862; macro-F1 0,785 (com fundo) | Partição espacial real (7 áreas treino / 6 teste) |
| Zheng et al. 2024 (CREVNet) | Satélite óptico + altimetria | F1 0,804 | In-domain; partição não declarada |
| Zhao et al. 2022 | Sentinel-1 SAR, 40 m | F1 0,725 | In-domain |
| Surawy-Stepney et al. 2023 | Sentinel-1 SAR, 50 m | AUC-ROC 0,91–0,93 (sem F1 por pixel) | 3 quadros independentes |

Três observações contextualizam o resultado. (i) Aplicando a correção empírica de Kattenborn et al. (2022) para o vazamento de particionamento aleatório (+14,6–27,9% de F1), nosso 0,654 espacial equivale a ~0,75–0,84 sob o protocolo dos estudos in-domain — exatamente a faixa de CREVNet e Duan. Reforço direto: a **U-Net clássica avaliada pelos próprios Duan et al. (2025) no protocolo deles atinge F1 0,765** (Tabela 5-1 do artigo) — em alvo de alto contraste (gelo seco tibetano, "sharp grayscale transitions"), a 3 cm, com partição por patch de janela deslizante sobre duas ROIs contínuas do mesmo terminus, o cenário exato de vazamento por vizinhança medido por Kattenborn et al. (2022). (ii) O único estudo com partição espacial comparável (Baumhoer et al., 2025) evita reportar F1 da classe fenda; sua acurácia balanceada de 0,862 inclui o fundo (~0,98), e o F1 implícito da classe fica abaixo de ~0,6. (iii) A tolerância de 4 px (~32 cm) é ~11× mais estrita, em termos físicos, que a folga-padrão de 3 px (~3,6 m) da literatura de extração de estradas, cujos valores de referência (0,89–0,92) só existem com essa folga.

\newpage

# 4. Resultados: evolução das fendas e ablação (2016–2019)

## 4.1 Produtos cartográficos e inventário

![](../results/figures/relatorio/mapa_fendas_2019_qgis.png){width=16cm}

*Figura 6 — Mapa das fendas supraglaciais segmentadas em 2019 (U-Net ep.107, TTA, blend probabilístico, limiar 0,6) sobre o ortomosaico de 8 cm, com contornos de elevação de 50 m e limite do glaciar. Recortes (a–d) em resolução nativa: (a) zona terminal junto ao lago proglacial, (b) campo denso central, (c) fraturas em superfície de neve da porção média, (d) fraturas finas do setor leste. n = 17.500 componentes; cobertura 1,18%; EPSG:32719 (UTM 19S).*

![](../results/figures/relatorio/mapa_mascaras_4anos_qgis.png){width=16cm}

*Figura 7 — Fendas segmentadas sobre os ortomosaicos de 8 cm, 2016–2019, com zoom fixo (campo central, 158 × 96 m) em resolução nativa evidenciando a fragmentação progressiva das feições. Máscaras em vermelho; limite do glaciar em preto.*

**Tabela 4 — Inventário anual de fendas (produtos finais; grade de 8 cm).**

| Ano | n componentes | n fendas (geometria filtrada) | Área total (10³ m²) | Cobertura (%) | Área mediana (m²) | Compr. mediano (m) | Altitude mediana (m) |
|---|---|---|---|---|---|---|---|
| 2016 | 11 551 | 10 083 | 59,9 | 1,40 | 1,95 | 6,4 | 132,6 |
| 2017 | 13 297 | 11 526 | 52,3 | 1,22 | 1,41 | 5,3 | 141,0 |
| 2018 | 14 885 | 12 939 | 63,5 | 1,48 | 1,52 | 5,6 | 138,9 |
| 2019 | 17 500 | 14 131 | 50,6 | 1,18 | 0,65 | 3,8 | 129,9 |

*A cobertura anual é sensível ao limiar de binarização (0,6) e às condições de superfície de cada voo; as comparações interanuais robustas são as de contagem por classe de tamanho, geometria e distribuição espacial (Fig. 12–14).*

![](../results/figures/relatorio/fig12_evolucao_anual.png){width=12cm}

*Figura 14 — Série anual (n = 4): contagem, área total, cobertura e altitude mediana, com envelope de incerteza derivado da precisão/revocação da validação (aproximação conservadora).*

## 4.2 Fragmentação e morfometria

![](../results/figures/relatorio/fig10_morfometria.png){width=15cm}

*Figura 12 — Distribuições de área e comprimento (bins logarítmicos), ECDF e larguras por ano.*

Entre 2016 e 2019 a contagem de fendas cresce 40% enquanto a área total cai 16%: o crescimento vem **inteiramente das classes pequenas** (fendas < 1 m² passam de 2 984 para 6 554 — de 29,6% para 46,4% da população; χ² = 958, p ≈ 10⁻²⁰⁷), enquanto as classes > 5 m² perdem 745 feições e ~12,8 mil m². As dimensões medianas encolhem sistematicamente (área −57%: 2,72 → 1,18 m²; comprimento −41%; largura −29%; KS p < 10⁻¹⁵⁰; δ de Cliff 0,22–0,25), com razão de aspecto estável (~10 → 9). As fendas grandes de 2019 são majoritariamente persistentes de 2016, enquanto as pequenas combinam fraturamento novo e subdivisão — no conjunto, **fragmentação progressiva da rede**.

## 4.3 Migração, persistência e orientação

![](../results/figures/relatorio/fig08_hipsometria.png){width=13cm}

*Figura 10 — dH, densidade de fendas e distribuição de elevação das fendas por banda de 50 m.*

![](../results/figures/relatorio/fig11_rosetas_orientacao.png){width=15cm}

*Figura 13 — Orientação axial das fendas por ano (rosetas espelhadas, raio ∝ √frequência).*

- **Expansão para montante:** 55–63% das fendas de 2019 (conforme critério de correspondência) não têm correspondente de 2016 e concentram-se a montante — elevação mediana ~18 m maior e ~250 m mais distantes do terminus que as persistentes (Mann-Whitney, p ≈ 10⁻¹²¹). A elevação mediana do conjunto cai apenas 2,7 m enquanto a superfície afinou ~9,5 m acumulados — a produção de fendas novas a montante compensa o rebaixamento.
- **Renovação do campo:** apenas 5,5% (sobreposição estrita) a 30,3% (tolerância ~1 m) da área de fendas de 2019 já era fenda em 2016 — renovação quase completa em 3 anos, combinando advecção (não modelada) e abertura de novas fraturas.
- **Orientação:** a direção axial dominante ENE (65–66°) persiste, com rotação leve para 70° e dispersão significativa em 2019 (IC95 da queda de concentração exclui zero) — consistente com reorganização do campo de tensões durante a retração. O aumento da declividade nas fendas (mediana 20,9° → 22,9°) acompanha o aumento da declividade da superfície como um todo.

## 4.4 Ablação e acoplamento fendas–afinamento (atividade 3)

![](../results/figures/relatorio/mapa_dh_qgis.png){width=16cm}

*Figura 9 — dH por diferenciação de MDEs (pares 2016–17, 2017–18, 2018–19), grade de 2 m, com contornos de elevação e domínio de gelo delimitado. Co-registro validado em terreno estável (|dH| ≤ 0,04 m).*

![](../results/figures/relatorio/mapa_densidade_qgis.png){width=16cm}

*Figura 8 — Densidade de fendas em blocos de 100 m por ano, persistência interanual e variação 2016→2019.*

**Tabela 5 — Correlação densidade de fendas × dH em blocos de 100 m (domínio de gelo).** *Valores não significativos (p ≥ 0,05) em itálico.*

| Par | n blocos | ρ Spearman | p | ρ parcial (elev.) | p parcial | Efeito (top-quartil − sem fenda) |
|---|---|---|---|---|---|---|
| 2016→17 | 215 | **−0,58** | 2×10⁻²⁰ | **−0,39** | 4×10⁻⁹ | **−1,97 m** (p = 7×10⁻¹¹) |
| 2017→18 | 217 | +0,19 | 0,004 | +0,25 | 2×10⁻⁴ | *+0,27 m (p = 0,17)* |
| 2018→19 | 216 | *−0,01* | *0,91* | *+0,02* | *0,80* | *−0,11 m (p = 0,86)* |

**Tabela 6 — dH por par de anos e validação de co-registro.**

| Par | dH mediano no gelo (m) | dH mediano, área total sobreposta (m) | dH em terreno estável (m) | n blocos estáveis |
|---|---|---|---|---|
| 2016→17 | −5,31 | −4,50 | +0,04 | 61 |
| 2017→18 | −3,25 | −2,47 | +0,01 | 65 |
| 2018→19 | −3,55 | −2,49 | −0,01 | 67 |

![](../results/figures/relatorio/fig09_scatter_densidade_dh.png){width=16cm}

*Figura 11 — Densidade de fendas × dH em blocos de 100 m, por par de anos, com ρ, p e n anotados. Os resultados nulos são mostrados com o mesmo desenho — o caráter episódico do acoplamento é o achado.*

No ano de afinamento máximo (2016→17), blocos mais fendilhados afinaram significativamente mais: gelo sem fendas perdeu −4,25 m contra −6,21 m no quartil superior de densidade — **um afinamento extra de ~2 m associado ao fendilhamento**, robusto ao controle de elevação e à escala de bloco (100–200 m). Nos anos de afinamento moderado o acoplamento desaparece (2018→19) ou inverte fracamente (2017→18), sugerindo modulação climática do mecanismo. Em bandas altitudinais a correlação some (ρ agregado = +0,06): a variabilidade relevante da densidade é **transversal ao fluxo**, na escala de centenas de metros — um resultado metodológico em si, pois o desenho de análise mais comum (bandas de elevação) não detectaria o acoplamento. Este conjunto endereça diretamente a lacuna apontada por Colgan et al. (2016).

## 4.5 Exploração dinâmica: velocidade, deformação e profundidade (testes de viabilidade)

Dois testes derivados de Duan et al. (2025) delimitam o que os dados atuais permitem. **(1) Feature tracking anual** (2018→2019; correlação cruzada normalizada por chips de 128 m): com co-registro validado em terreno estável (deslocamento mediano de 0,00 m), **79% da superfície do gelo decorrelaciona** no intervalo anual — um campo completo de velocidade é inviável com pares anuais —, mas os 173 vetores aprovados no controle de qualidade são fisicamente consistentes: 77% apontam para SW/W (direção do lago proglacial), com velocidade mediana de ~91 m a⁻¹ e a língua terminal mais lenta (~40 m a⁻¹) que a porção a montante (~93 m a⁻¹), padrão coerente com estagnação do terminus em afinamento acelerado. O campo de deformação derivado (Nye, 1959) passa no teste de mecânica de fratura de Harper et al. (1998): o ângulo mediano entre o eixo das fendas e o eixo da deformação principal ε₁ é de **73,6° (74% acima de 60°; perpendicularidade esperada ~90°)** — uma validação física independente da orientação das fendas segmentadas.

![](../results/figures/relatorio/fig15_velocidade.png){width=16cm}

*Figura 15 — Velocidade superficial 2018→2019 por feature tracking. (a) Vetores aprovados no controle de qualidade (n = 173) sobre hillshade, coloridos pela magnitude; a superfície restante (79%) decorrelaciona no intervalo anual. (b) Velocidade × posição ao longo do eixo de fluxo (proxy O→L; a frente e o lago proglacial ficam a oeste): o fluxo ativo acelera de ~85 para ~145 m a⁻¹ para montante, enquanto um aglomerado de vetores quase nulos junto à frente indica gelo estagnado/morto — coerente com o afinamento e a fragmentação observados. Coordenadas EPSG:32719.* **(2) Profundidade aparente** por ajuste de plano de referência local (adaptação fotogramétrica do workflow LiDAR de Duan): o MDE SfM de ~31 cm não penetra as fendas — profundidade aparente mediana de 0,14–0,17 m (máximo 0,41 m; ~90% dos componentes reprovados no controle de qualidade do plano), uma ordem de grandeza abaixo do obtido com LiDAR embarcado (3,06 ± 3,91 m) — estabelecendo quantitativamente que a extração de profundidade 3D requer LiDAR (trabalho futuro).

\newpage

# 5. Discussão e limitações

**O modelo opera no teto de consistência do próprio ground truth.** Um experimento de consistência intra-anotador (re-anotação às cegas de 15 tiles estratificados, ≥2 meses após a anotação original, originais ocultados durante a sessão) resultou em auto-concordância de **F1 = 0,583 (estrita)** e **0,785 (tolerância de 4 px)**. Na métrica estrita, o F1 out-of-sample do modelo (0,654) **supera a concordância do anotador consigo mesmo**; sob tolerância de 4 px, o modelo concorda com o GT (0,899 nos mesmos 15 tiles; 0,833 nos 218 — valores in-sample) mais do que o anotador concorda com a própria anotação. A divergência humana é simétrica (P ≈ R ≈ 0,58) e concentra-se na largura/halo das feições — o mesmo modo de "erro" dominante do modelo. A implicação metodológica é direta: ganhos adicionais de F1 pixel-wise ajustariam ruído de anotação, não estrutura real, o que fundamenta a decisão de congelar a otimização do modelo e priorizar as análises glaciológicas. *(Ressalvas: n = 15; um único anotador — a concordância inter-anotador, protocolo mais forte, está planejada para a publicação do conjunto de dados.)*

**O resultado de segmentação é aceitável e competitivo quando o protocolo é considerado.** A comparação nua de F1 entre estudos é enganosa (Tabela 3): os valores mais altos da literatura provêm de avaliações in-domain com particionamento não declarado (e, no caso mais alto, com LiDAR adicional), enquanto o único estudo com partição espacial real evita o F1 por classe. Sob a conversão empírica de Kattenborn et al. (2022), nosso resultado espacial equivale à faixa in-domain publicada; na única métrica com comparável direto (AUC-ROC), igualamos os valores da The Cryosphere (0,93/0,91).

**Limitações quantificadas:**

- *Co-registro dos MDEs:* sem ICP; viés vertical medido ≤ |0,04| m em terreno estável (mitigado).
- *Advecção não modelada:* "fenda nova" e "persistência" são conceitos eulerianos; com fluxo de ~3 anos, a persistência lagrangiana real é subestimada (por isso reportamos a faixa 5,5–30,3%).
- *GT interanual:* carrega viés do modelo anterior (Seção 2.2); as métricas de transferência são indicativas, e a validação primária é o CV espacial de 2016.
- *Sensibilidade ao limiar:* comparações de cobertura entre anos variam com o limiar (0,6); as conclusões estruturais (fragmentação, migração, acoplamento) são robustas — reproduzem-se em três gerações sucessivas do modelo.
- *Limite de detecção:* componentes < ~0,5 m² não são detectados (respondem por apenas 8% dos falsos negativos em área). A decomposição da distância de Hausdorff confirma o diagnóstico: o HD95 restrito às feições detectadas é de ~10–21 px (0,8–1,7 m), enquanto o HD95 bruto (44–90 px) é dominado por fendas finas inteiras não detectadas — o erro de cauda é de completude de detecção, não de delineação de borda.
- *Autocorrelação espacial:* os p-valores por bloco são otimistas (n efetivo < n blocos); por isso reportamos sempre o tamanho de efeito em metros junto ao p.
- *Série curta:* n = 4 anos não permite tendências estatísticas; as séries anuais são descritivas.

# 6. Conclusões

1. Uma U-Net de seis canais treinada com um único ano de dados atinge **F1 espacial de 0,654 ± 0,014** na segmentação de fendas em VANT de 8 cm, com transferência temporal estável (0,66–0,72) e precisão de ~0,90 sob tolerância de 32 cm — desempenho competitivo com a literatura quando normalizado pelo rigor do protocolo, constituindo o primeiro benchmark do gênero.
2. A **auditoria de rótulos** é decisiva nesse regime de dados: 30% dos negativos continham fendas reais e sua correção valeu +5 p.p. de precisão — mais que qualquer ajuste de arquitetura ou hiperparâmetro testado.
3. O campo de fendas do Glaciar Schiaparelli **fragmenta-se e migra para montante** (2016–2019): +40% em número às custas das classes pequenas, −16% em área, renovação quase completa em 3 anos e fendas novas ~250 m mais distantes do terminus.
4. **O fendilhamento acopla-se ao afinamento de forma episódica**: no ano de ablação máxima, gelo fendilhado afinou ~2 m a mais que gelo sem fendas (ρ parcial = −0,39), com o acoplamento ausente em anos moderados e invisível em análises por banda altitudinal — evidência quantitativa, em escala de 100–200 m, da retroalimentação fendas–ablação.

**Trabalhos futuros:** publicação do conjunto de dados como benchmark; teste-set independente em 2020; perdas topológicas (clDice) e métricas de esqueleto (clIoU); campo de velocidade completo por pares de curto intervalo (intra-campanha) ou tracking manual de feições persistentes (a decorrelação anual de 79% inviabiliza o tracking automático denso); LiDAR embarcado para profundidade 3D das fendas (a fotogrametria mede apenas ~0,15 m de profundidade aparente); extensão a lagos e canais supraglaciais.

# 7. Atividades realizadas × cronograma

| Atividade do edital | Status |
|---|---|
| 1. Treinamento Agisoft Metashape / dados processados | Concluída |
| 2. Processamento VANT (tiles 2016–2020, normalização 8 cm) | Concluída |
| 3. Estimativa de ablação superficial (dH + correlação densidade×dH) | **Concluída** |
| 4. Mapeamento por aprendizado de máquina (fendas 2016–2019) | **Concluída** (lagos/canais como extensão futura) |

Marcos técnicos principais: transição SAM→U-Net (abr/2026); modelo 6 canais e augmentação cromática (mai/2026); normalização de resolução e análises em 3 camadas (jun/2026); validação cruzada espacial (jun/2026); auditoria de rótulos, correção do pipeline de reconstrução e produtos finais (ago/2026).

# Referências

- Baumhoer, C. et al. (2025). Automated crevasse mapping for Alpine glaciers. *Int. J. Appl. Earth Obs. Geoinf.* 139:104495.
- Colgan, W. et al. (2016). Glacier crevasses: observations, models, and mass balance implications. *Reviews of Geophysics* 54.
- Duan, Y. et al. (2025). Automated high-resolution 3D crevasse extraction: C-TransUNet. *Int. J. Appl. Earth Obs. Geoinf.* 144:104881.
- Hugonnet, R. et al. (2021). Accelerated global glacier mass loss in the early twenty-first century. *Nature* 592.
- Kattenborn, T. et al. (2022). Spatially autocorrelated training and validation samples inflate performance assessment of CNNs. *ISPRS Open J. Photogramm. Remote Sens.* 5:100018.
- Kneib, M. et al. (2021). Mapping ice cliffs on debris-covered glaciers. *Remote Sens. Environ.* 253:112201.
- Mnih, V.; Hinton, G. (2010). Learning to detect roads in high-resolution aerial images. *ECCV*.
- Ploton, P. et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications* 11.
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40:913–929.
- Shit, S. et al. (2021). clDice — a novel topology-preserving loss for tubular structure segmentation. *CVPR*.
- Surawy-Stepney, T. et al. (2023). Mapping Antarctic crevasses using deep learning. *The Cryosphere* 17:4421–4445.
- Benz, C.; Rodehorst, V. (2024). OmniCrack30k: a benchmark for crack segmentation. *CVPR Workshops*.
- Wallace, S. et al. (2025). Exploring Segment Anything foundation models for out-of-domain crevasse drone image segmentation. *NLDL, PMLR* 265:255–268.
- Zhao, J. et al. (2022). Detection of surface crevasses over Antarctic ice shelves. *Remote Sensing* 14(3):487.
- Zheng, L. et al. (2024). CREVNet. *IEEE Geosci. Remote Sens. Lett.*, doi:10.1109/LGRS.2024.3407860.
- Wiedemann, C. et al. (1998). Empirical evaluation of automatically extracted road axes. *CVPR Workshops*.
