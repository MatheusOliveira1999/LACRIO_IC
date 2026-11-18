---
title: ""
lang: pt-BR
---

# Relatório Final de Iniciação Científica

**Mudanças na superfície de geleiras a partir de dados VANT: mapeamento automático de fendas e sua relação com a perda de gelo no Glaciar Schiaparelli (Cordilheira Darwin, Chile)**

| | |
|-------------------------|---------------------------------------------------------------------------|
| **Bolsista** | Matheus Oliveira |
| **Orientador** | Prof. Dr. Jorge Arigony Neto |
| **Laboratório** | LaCrio — Laboratório de Monitoramento da Criosfera, FURG |
| **Vigência da bolsa** | Setembro/2025 – Agosto/2026 (CNPq) |
| **Data do relatório** | Agosto de 2026 (revisão de setembro de 2026) |

## Resumo

Este projeto criou um sistema de inteligência artificial que encontra e desenha automaticamente as fendas (*crevasses*) nas fotos de drone do Glaciar Schiaparelli, na Terra do Fogo, Chile, com pixels de 8 cm. Com os mapas produzidos, medimos pela primeira vez neste glaciar se as áreas com mais fendas perdem mais gelo.

O modelo é uma rede neural do tipo U-Net que recebe seis camadas de informação: as três cores da foto e três camadas tiradas do mapa de altura do terreno (relevo, inclinação e curvatura). Ele foi treinado apenas com as fotos de 2016 e testado em regiões do glaciar que nunca viu durante o treino. Nesse teste exigente, a nota F1 (de 0 a 1, que resume acertos e erros) foi **0,66 ± 0,02**, e ficou na mesma faixa (0,66 a 0,71) quando o modelo foi aplicado às fotos de 2017, 2018 e 2019 sem novo treino.

O que mais melhorou o resultado não foi mexer no modelo, e sim corrigir o gabarito. Uma revisão mostrou que quase metade dos recortes tratados como "sem fenda" sem conferência (32 de 65) tinha fendas; corrigi-los aumentou a precisão em 5 pontos. Mudanças no treino (outras formas de calcular o erro, limpeza da resposta, treino com fotos não anotadas) não trouxeram ganho. O único ganho repetível veio de juntar dez modelos em um comitê, o que elevou o F1 nos outros anos para 0,70.

Os mapas de 2016 a 2019 mostram três coisas. Primeiro, no ano de maior perda de gelo (2016→17), os quadrados de 100 m com mais fendas baixaram cerca de **2 m a mais** que o gelo sem fendas; essa ligação não aparece nos anos de perda moderada. Segundo, as fendas ficaram mais numerosas (+40%) e menores (área total −16%): a rede de fendas está se **fragmentando**. Terceiro, as fendas novas de 2019 surgem **mais acima** no glaciar do que as antigas (cerca de 18 m mais alto e 250 m mais longe da frente).

Por fim, quando o anotador remarcou às cegas recortes que já tinha marcado, a concordância consigo mesmo foi 0,58 (0,79 com folga de 32 cm), abaixo do modelo (0,66 e 0,85). O modelo chegou ao limite do que o próprio gabarito permite medir. Os recortes anotados formam um dos primeiros conjuntos de fendas em fotos de drone, e o primeiro testado em regiões separadas e com quatro anos de série.

**Palavras-chave:** fendas; drone (VANT); inteligência artificial; U-Net; validação espacial; perda de gelo; Cordilheira Darwin.

## Termos usados neste relatório

- **Fenda (*crevasse*):** rachadura que se abre na superfície da geleira quando o gelo se estica ao escorrer.
- **VANT:** veículo aéreo não tripulado, ou seja, o drone.
- **Ortomosaico:** uma única imagem do glaciar montada a partir de centenas de fotos do voo.
- **MDE (modelo digital de elevação):** mapa em que cada pixel guarda a altura do terreno.
- **Recorte:** pedaço de 512 × 512 pixels do ortomosaico (cerca de 41 × 41 m). O modelo analisa um recorte por vez.
- **Gabarito:** as fendas marcadas à mão. Serve para ensinar o modelo e para corrigir as respostas dele.
- **Precisão:** de tudo que o modelo marcou como fenda, quanto era fenda de verdade.
- **Revocação:** de todas as fendas do gabarito, quanto o modelo encontrou.
- **F1:** nota de 0 a 1 que junta precisão e revocação. Só é alta se as duas forem altas.
- **Limiar:** o modelo dá a cada pixel uma chance de 0 a 1 de ser fenda. O limiar é o valor a partir do qual o pixel é pintado como fenda.
- **Validação cruzada espacial:** o glaciar é dividido em blocos; o modelo é treinado sem um grupo de blocos e testado justamente nele. Assim a nota mede o acerto em lugares nunca vistos.
- **Folga de 4 pixels (32 cm):** forma de medir em que errar a borda da fenda por até 32 cm não conta como erro.
- **Comitê de modelos (*ensemble*):** a resposta final é a média das respostas de vários modelos.
- **dH:** quanto a superfície baixou ou subiu entre dois anos, em metros. Valor negativo significa perda de gelo.
- **Correlação (ρ):** número de −1 a +1 que diz se duas coisas variam juntas. Aqui, ρ negativo quer dizer "mais fendas, mais perda de gelo".
- **Valor p:** chance de um resultado aparecer só por acaso. Abaixo de 0,05, considera-se que não é acaso.

# 1. Introdução e objetivos

Fendas são rachaduras que se abrem na superfície de uma geleira quando o gelo se estica ao escorrer encosta abaixo. Elas mostram como o gelo está se movendo e também ajudam a derretê-lo: deixam a superfície mais rugosa, aumentam a área exposta ao sol e ao ar e servem de caminho para a água de degelo. Colgan et al. (2016), ao reunirem o que se sabe sobre fendas, descreveram esses mecanismos. Medir o efeito com números, região por região, continua raro, porque marcar todas as fendas à mão é lento demais e os satélites não enxergam as pequenas.

As fotos de drone mudam esse cenário, porque cada pixel mede poucos centímetros no chão e até fendas finas aparecem. O que falta é um jeito automático e confiável de marcá-las. Até a nossa busca de agosto de 2026, não encontramos nenhum conjunto público de fotos de drone de geleiras com fendas anotadas^[Casos verificados em 09/08/2026: (i) Wallace et al. (2025) anotaram 10 imagens de drone (Borebreen, Svalbard), mas o repositório tem apenas código e as anotações não foram publicadas; (ii) fotos de drone de geleiras com fendas já são públicas, porém sem anotações (Hann et al., 2024; DataverseNO) — o que falta são as máscaras, não as fotos; (iii) nos Alpes há mapas públicos de fendas feitos com avião tripulado a 20 cm (Baumhoer et al., 2025), mas são respostas de um modelo, sem gabarito; (iv) existem produtos públicos de fendas em imagens de satélite (Zenodo 7254130, 6779087); (v) Duan et al. (2025) informam que os dados estão disponíveis apenas "sob pedido". Foram consultados Zenodo, DataverseNO/Harvard Dataverse, Figshare, Mendeley Data, IEEE DataPort, PANGAEA, Papers With Code, Roboflow, HuggingFace, Kaggle, GitHub, arXiv, OpenAlex e DOAJ, com buscas em inglês e chinês. **Atualização de setembro de 2026:** um preprint de 16/08/2026 (Wallace et al., 2026, "CrevasseSeg") anota cerca de 200 recortes de drone do mesmo glaciar de Svalbard e anuncia que vai publicá-los; até 18/09/2026 o repositório de dados não havia sido localizado.]. O único estudo publicado com notas nesse tipo de foto (Wallace et al., 2025) usou um modelo genérico, o SAM 2, sem treiná-lo para fendas, e obteve acerto fraco (0,43 em uma escala de 0 a 1). Os próprios autores apontam a necessidade de modelos treinados especificamente para essa tarefa.

O plano da bolsa previa quatro atividades: (1) processar as fotos no Agisoft Metashape; (2) preparar os levantamentos de drone de 2016 a 2022; (3) estimar a perda de gelo na superfície; e (4) mapear feições da superfície com aprendizado de máquina. A partir delas, o projeto teve quatro objetivos:

1. Criar e testar um modelo que marque fendas automaticamente nas fotos de drone de 8 cm do Glaciar Schiaparelli;
2. Produzir mapas de fendas de 2016 a 2019, com uma forma de avaliação rigorosa;
3. Medir quanto a superfície baixou a cada ano e **verificar se as áreas com mais fendas perderam mais gelo** (atividade 3);
4. Descrever como a rede de fendas mudou ao longo do tempo (fragmentação, deslocamento e direção) como parte da análise integrada (atividade 4).

# 2. Dados e métodos

## 2.1 Área de estudo e dados

O Glaciar Schiaparelli fica na Cordilheira Darwin, Terra do Fogo, Chile (cerca de 54,4°S, 70,9°W). Ele foi fotografado por drone uma vez por ano, de 2016 a 2020, e as fotos foram processadas no Agisoft Metashape (atividade 1). Cada ano gerou dois produtos: um ortomosaico colorido, com pixels de 5,4 a 8,2 cm, e um mapa de altura (MDE), com pixels de 22 a 31 cm. Para comparar os anos de forma justa, os ortomosaicos de 2016 a 2019 foram levados a uma mesma grade de **8 cm por pixel** e cortados em cerca de 2.200 recortes por ano, com uma pequena sobreposição entre vizinhos (atividade 2). A área coberta é de cerca de 4,3 km² (Figura 1).

```{=openxml}
<w:p><w:pPr><w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1418" w:right="1418" w:bottom="1418" w:left="1418" w:header="709" w:footer="709" w:gutter="0"/></w:sectPr></w:pPr></w:p>
```

![](../results/figures/relatorio/mapa_area_estudo_qgis.png){height=15cm}

*Figura 1 — Localização da área de estudo: Glaciar Schiaparelli, Cordilheira Darwin (Terra do Fogo, Chile; cerca de 54,40°S, 70,88°W). Mapa principal: ortomosaico do drone de 2016 (8 cm por pixel), com curvas de nível a cada 50 m e o limite do glaciar. Quadros menores: localização no continente e na região (mapa-base © OpenStreetMap/CARTO). Coordenadas UTM zona 19S.*

```{=openxml}
<w:p><w:pPr><w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/><w:pgMar w:top="851" w:right="851" w:bottom="851" w:left="851" w:header="425" w:footer="425" w:gutter="0"/></w:sectPr></w:pPr></w:p>
```

## 2.2 O gabarito e a revisão das anotações

As fendas de 2016 foram marcadas à mão, pixel por pixel, com uma ferramenta criada no próprio projeto (`annotate.py`) e uma regra clara. Contam como fenda as linhas escuras e alongadas, que aparecem em famílias paralelas e têm sombra profunda. Não contam ondulações da neve, canais de água nem sombras de rochas. O conjunto final tem **218 recortes de 2016: 144 com fendas e 74 conferidos sem fenda**.

Ao longo do projeto encontramos três problemas no gabarito. Eles foram corrigidos e fazem parte do método:

- **Recortes "sem fenda" que não tinham sido conferidos.** No início, o treino completava o conjunto com recortes escolhidos ao acaso e tratados como "sem fenda" sem ninguém olhar (30% do conjunto). Esses 65 recortes concentravam 20,4% de tudo que o modelo estava sendo "punido" por marcar. Revisados um a um, 32 deles tinham fenda de verdade e 33 realmente não tinham. Com a correção, a precisão medida subiu 5 pontos. Desde então só entram recortes "sem fenda" já conferidos.
- **Revisão final (setembro de 2026).** Os 18 recortes que ainda estavam em dúvida foram remarcados à mão (14 com fenda, 4 sem), fechando o conjunto em 144 + 74. Essa correção mudou a *medição*, não o modelo: os mesmos modelos, avaliados no gabarito corrigido, passaram de F1 0,654 para 0,660. Treinar um modelo novo com o gabarito corrigido deu o mesmo resultado (diferença de +0,1 ponto, que pode ser acaso; p = 0,77).
- **Gabarito dos outros anos contaminado pelo modelo antigo.** As anotações de 2017, 2018 e 2019 (38, 39 e 40 recortes) foram revisadas partindo da resposta de um modelo anterior, e em cerca de 40% dos recortes ficaram idênticas a essa resposta. Por isso, notas antigas de F1 entre 0,83 e 0,87 foram descartadas: o modelo estava sendo comparado com as próprias respostas. Os números desses anos neste relatório vêm com esse aviso.

## 2.3 O modelo e o treino

O modelo é uma **U-Net**, um tipo de rede neural que classifica a imagem pixel por pixel. A parte que "lê" a imagem (ResNet-34) já vem treinada em milhões de fotos comuns. Além das três cores da foto, o modelo recebe três camadas calculadas do mapa de altura: o relevo local, a inclinação e a curvatura. Isso ajuda a separar uma fenda, que é um sulco, de uma mancha escura de pedras, que é plana.

Durante o treino, as fotos passam por variações artificiais (giros, espelhamentos e mudanças leves de brilho e cor, inclusive o escurecimento de um lado da imagem, como acontece em voos com luz desigual). Isso ensina o modelo a não depender da iluminação de um dia específico. A forma de calcular o erro pune mais os falsos alarmes do que as fendas perdidas. O treino durou 200 rodadas; o melhor modelo saiu na rodada 107, depois de cerca de uma hora em uma placa de vídeo comum (GTX 1650). Esse é o modelo usado nos mapas. Os dez modelos criados nas duas validações cruzadas (Seção 2.4) formam o **comitê** avaliado na Seção 3.4.

## 2.4 Como o acerto foi medido

A nota principal vem de uma **validação cruzada espacial** (Roberts et al., 2017). O glaciar foi dividido em blocos de cerca de 160 m, e os blocos foram repartidos em cinco grupos. O modelo foi treinado cinco vezes; em cada vez, um grupo inteiro ficou de fora e serviu de teste (Figura 2). Além disso, qualquer recorte de treino que encostasse em um recorte de teste foi removido. Isso importa porque recortes vizinhos são muito parecidos: quando o teste fica ao lado do treino, a nota sobe artificialmente. Em fotos de drone, esse efeito já foi medido em 15% a 28% a mais de F1 (Kattenborn et al., 2022).

Também medimos: (a) o acerto nos anos de 2017 a 2019, com o modelo treinado só em 2016 e sem novo treino; e (b) o acerto com folga de 4 pixels (32 cm), que é o costume para feições finas, como rachaduras e estradas. Usamos duas versões dessa medida: a do conjunto OmniCrack30k (Benz e Rodehorst, 2024), que compara as linhas centrais das feições, e a precisão/revocação "com folga" usada para estradas (Wiedemann et al., 1998; Mnih e Hinton, 2010).

Na produção dos mapas, cada recorte é analisado em **quatro orientações** e as respostas são somadas (ganho de +0,6 ponto de F1 e +1,5 de precisão, repetido nos cinco grupos; p = 0,007). O **limiar** usado é 0,6, que fica no meio da faixa em que o F1 quase não muda (Figura 3). Os recortes vizinhos são **costurados dando mais peso ao centro** de cada um, onde o modelo acerta mais. Isso substituiu a costura antiga, que pintava um pixel como fenda se qualquer recorte o marcasse: nas faixas de sobreposição (25,8% da área), a precisão da costura antiga caía para 0,39. A nova costura aumentou a precisão do mapa final em 1,7 ponto.

Toda comparação entre duas versões do modelo foi feita **recorte a recorte**, nos mesmos 218 recortes de teste, com intervalo de confiança calculado por reamostragem (2.000 repetições) e teste de Wilcoxon (`17_compare_oof.py`). A validação cruzada foi repetida com uma segunda divisão dos blocos, para ver se o resultado dependia da forma de dividir. O **comitê** é a média das respostas dos dez modelos dessas duas validações. Ele só foi avaliado em 2017–2019, porque em 2016 cada modelo do comitê já viu 80% dos recortes.

## 2.5 Análises do glaciar

A perda de gelo (atividade 3) foi medida subtraindo o mapa de altura de um ano do mapa do ano seguinte, numa grade comum de 2 m. Para conferir o alinhamento entre os mapas, medimos a diferença em terreno que não deveria mudar (rocha): ela ficou em no máximo 4 cm, cem vezes menor que a perda medida no gelo. A **relação entre fendas e perda de gelo** foi calculada em quadrados de 100 m sobre o gelo. Usamos a correlação de Spearman, uma segunda versão que **desconta o efeito da altitude**, e a comparação direta entre o gelo sem fendas e os 25% de quadrados com mais fendas (teste de Mann-Whitney). A área de gelo foi definida com dados de outros anos, para não influenciar o resultado. O script é `11_density_dh_analysis.py`. As demais análises (contagem, tamanho, direção, persistência e altitude das fendas) usam os mapas finais de 8 cm e os mapas de altura.

```{=openxml}
<w:p><w:pPr><w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1418" w:right="1418" w:bottom="1418" w:left="1418" w:header="709" w:footer="709" w:gutter="0"/></w:sectPr></w:pPr></w:p>
```

![](../results/figures/relatorio/mapa_folds_cv.png){height=15cm}

*Figura 2 — Como o glaciar foi dividido para o teste. Cada quadradinho é um dos 218 recortes anotados de 2016: cheio = com fenda, vazado = conferido sem fenda. A cor indica o grupo de teste, e não a presença de fenda: recortes com e sem fenda entram no teste, os primeiros para medir se o modelo acha as fendas e os segundos para medir se ele inventa fenda onde não há. Os recortes sem fenda sobre rocha e morena, fora do limite do glaciar, foram incluídos de propósito. À direita, um exemplo de recorte com fenda de cada grupo (número no mapa). Coordenadas UTM zona 19S.*

```{=openxml}
<w:p><w:pPr><w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/><w:pgMar w:top="851" w:right="851" w:bottom="851" w:left="851" w:header="425" w:footer="425" w:gutter="0"/></w:sectPr></w:pPr></w:p>
```

# 3. Resultados: o modelo

## 3.1 Acerto em regiões nunca vistas

**Tabela 1 — Acerto do modelo em 2016, testado em blocos nunca vistos (218 recortes; gabarito final de setembro de 2026; limiar 0,5).**

| Medida | O que significa | Valor (média ± variação entre os grupos) |
|----------------------------------|------------------------------------|------------------------------|
| Precisão | do que foi marcado, quanto era fenda | 0,627 ± 0,037 |
| Revocação | das fendas reais, quanto foi achado | 0,703 ± 0,036 |
| **F1** | nota que junta as duas | **0,661 ± 0,018** |
| AUC-PR | qualidade geral, para qualquer limiar (0 a 1) | 0,660 |
| AUC-ROC | capacidade de separar fenda de não fenda (0 a 1) | 0,935 |
| Precisão / revocação / F1 com folga de 32 cm | erro de borda de até 4 pixels não conta | 0,836 / 0,856 / **0,846** |
| clIoU com folga de 4 pixels (medida do OmniCrack30k) | compara as linhas centrais das fendas | 0,663 |
| F1 com outra divisão dos blocos | mostra se o resultado depende da divisão | 0,663 ± 0,030 |
| F1 analisando cada recorte em 4 orientações | como é feito nos mapas | 0,667 ± 0,019 (precisão 0,643) |

**Tabela 2 — Acerto em cada grupo de blocos e nos outros anos.**

| Teste | Recortes | Precisão | Revocação | F1 |
|--------------------------------------|--------------|----------------|----------------|----------------|
| Grupo 1 | 44 | 0,685 | 0,647 | 0,665 |
| Grupo 2 | 44 | 0,604 | 0,673 | 0,637 |
| Grupo 3 | 44 | 0,573 | 0,734 | 0,643 |
| Grupo 4 | 43 | 0,640 | 0,727 | 0,680 |
| Grupo 5 | 43 | 0,632 | 0,735 | 0,679 |
| 2016, nos recortes do próprio treino¹ | 218 | 0,694 | 0,745 | 0,718 |
| 2017, um modelo² | 38 | 0,674 | 0,684 | 0,679 |
| 2018, um modelo² | 39 | 0,669 | 0,751 | 0,708 |
| 2019, um modelo² | 40 | 0,672 | 0,656 | 0,664 |
| 2017, comitê de 10 modelos³ | 38 | 0,670 | 0,709 | 0,689 |
| 2018, comitê de 10 modelos³ | 39 | 0,647 | 0,798 | 0,714 |
| 2019, comitê de 10 modelos³ | 40 | 0,681 | 0,725 | 0,702 |

¹ *Nota otimista, porque o modelo já viu esses recortes; aparece só para comparar com estudos que testam assim.* ² *Modelo dos mapas, com 4 orientações e limiar 0,6. O gabarito desses anos tem o aviso da Seção 2.2.* ³ *Média dos dez modelos das duas validações, limiar 0,4 (o melhor para o comitê); mesmo gabarito.*

![](../results/figures/relatorio/fig02_curvas_prf1_threshold.png){width=11cm}

*Figura 3 — Precisão, revocação e F1 conforme o limiar escolhido (teste em blocos nunca vistos, gabarito final; a faixa sombreada mostra a variação entre os grupos; linha tracejada = 4 orientações). Entre 0,40 e 0,65 o F1 quase não muda, então o limiar pode ser escolhido para favorecer a precisão. O limiar dos mapas (0,6) está marcado.*

O F1 de 0,661 é estável entre os grupos (0,637 a 0,680) e entre as duas divisões dos blocos (0,663 na segunda). Nos outros anos o acerto não cai: fica entre 0,66 e 0,71 com um modelo e entre 0,69 e 0,71 com o comitê. Isso mostra que um modelo treinado com um único ano, usando as camadas de relevo e as variações de cor no treino, funciona em superfícies bem diferentes: neve fresca em 2017 e gelo exposto em 2018 e 2019.

## 3.2 Exemplos e tipos de erro

![](../results/figures/relatorio/fig03_qualitativa.png){width=15cm}

*Figura 4 — Exemplos de recortes (foto | gabarito | resposta do modelo | sobreposição): (a) fendas largas com detritos, F1 = 0,91; (b) rachaduras finas em gelo limpo, F1 = 0,90; (c) campo denso de rachaduras, F1 = 0,70; (d) fendas cobertas de neve, que o modelo perde com frequência (revocação 0,33); (e) blocos de gelo (seracs) e sombras (revocação 0,35). Gabarito em turquesa; resposta do modelo em azul.*

![](../results/figures/relatorio/fig04_falsos_positivos.png){width=15cm}

*Figura 5 — Tipos mais comuns de falso alarme: sombras do relevo, pedras e detritos escuros, canais de água e textura da neve. Mesmo assim, 68,6% dos pixels de falso alarme estão a no máximo 32 cm de uma fenda do gabarito: o erro mais comum é a largura da fenda, e não uma fenda inventada.*

## 3.3 Comparação com outros estudos

**Tabela 3 — Outros estudos de mapeamento de fendas. Os números não devem ser comparados diretamente: o tipo de imagem, a medida e a forma de teste mudam de um estudo para outro.**

| Estudo | Imagem | Resultado | Como foi testado |
|------------------|------------------|----------------------------------|------------------------------|
| **Este trabalho (2026)** | Drone, cores + relevo, 8 cm | **F1 0,661 ± 0,018**; AUC-ROC 0,935; F1 com folga 0,85; comitê nos outros anos F1 0,70 | **Blocos nunca vistos + 3 outros anos; gabarito revisado** |
| Wallace et al. 2025 | Drone, cores, 10 cm | DSC 0,43 (medida parecida com o F1) | Modelo genérico (SAM 2) sem treino para fendas; 10 imagens |
| Duan et al. 2025 (C-TransUNet) | Drone, cores, 3 cm | F1 0,871 (U-Net comum no mesmo teste: 0,765) | Teste em pedaços vizinhos do treino, na mesma área; nota provavelmente inflada |
| Baumhoer et al. 2025 | Avião, 20 cm | Acurácia balanceada 0,862 (inclui o fundo) | Áreas de teste separadas das de treino |
| Zheng et al. 2024 (CREVNet) | Satélite + altímetro | F1 0,804 | Mesma área; divisão não informada |
| Zhao et al. 2022 | Satélite radar, 40 m | F1 0,725 | Mesma área |
| Surawy-Stepney et al. 2023 | Satélite radar, 50 m | AUC-ROC 0,91–0,93 (sem F1) | Imagens independentes |

Três pontos ajudam a ler essa tabela. Primeiro, se aplicarmos a correção medida por Kattenborn et al. (2022) para testes feitos ao lado do treino (+15% a +28%), o nosso 0,661 equivaleria a cerca de 0,76 a 0,85 no formato de teste dos outros estudos, justamente a faixa de CREVNet e Duan. O próprio artigo de Duan et al. (2025) confirma isso: no teste deles, uma U-Net comum chega a 0,765, e isso em gelo seco de alto contraste, com pixels de 3 cm e teste em pedaços vizinhos do treino. Segundo, o único estudo com áreas de teste separadas (Baumhoer et al., 2025) não informa o F1 da classe fenda; a acurácia de 0,862 inclui o fundo, e o F1 das fendas fica provavelmente abaixo de 0,6. Terceiro, a nossa folga de 32 cm é cerca de 11 vezes mais rígida, em metros, que a folga comum nos estudos de estradas (cerca de 3,6 m). Na medida do OmniCrack30k, o nosso 0,66 fica no nível dos melhores valores daquele conjunto de rachaduras (0,64 do nnU-Net), que é de outra área.

## 3.4 O que muda e o que não muda o resultado

Depois da entrega dos mapas (agosto de 2026), fizemos uma série de testes controlados, sempre comparando recorte a recorte nos mesmos recortes de teste (Tabela 4). Só as mudanças feitas **na hora de usar o modelo** trouxeram ganho repetível. Nenhuma mudança no treino superou o controle, e duas delas pioraram o resultado.

**Tabela 4 — Testes feitos depois da entrega (2016, blocos nunca vistos, exceto quando indicado). Entre colchetes, o intervalo de confiança de 95%.**

| O que foi testado | Efeito no F1 | Decisão |
|--------------------------------------------|----------------------------------------|----------------|
| Analisar cada recorte em 4 orientações | +0,6 ponto [+0,2; +0,9], nos 5 grupos, p = 0,007; precisão +1,5 | adotado |
| Comitê de modelos (de 5 para 10), testado em 2017–19 | 0,680 → 0,698 (limiar 0,5) / 0,702 (limiar 0,4); precisão 0,626 → 0,704 | aprovado, candidato para os mapas |
| Limiar, tamanho mínimo das fendas e outras limpezas da resposta | até +0,1 ponto | descartado |
| 8 orientações em vez de 4 | +0,1 ponto | descartado |
| Função de erro clDice (Shit et al., 2021), treino do zero | −14 pontos (precisão 0,64 → 0,40) | descartado |
| 25 rodadas extras de treino, com e sem clCE (Acebes et al., 2024) ou clDice | 0,0 / +0,1 / +0,1 ponto (pode ser acaso) | sem efeito |
| clCE com treino do zero | −0,5 ponto [−1,2; +0,3]; com 4 orientações, −0,7 (piora real) | sem efeito ou pior |
| Treinar de novo com o gabarito corrigido | +0,1 ponto [−0,6; +0,8]; a correção em si vale +0,6 na medição | sem efeito |
| Treinar também com 936 recortes não anotados, rotulados pelo próprio modelo | +0,5 ponto [−1,1; +2,5] (pode ser acaso); precisão −4,9 | não adotado (troca precisão por revocação) |

Esses testes dizem três coisas. Primeiro, o comitê ganha porque os dez modelos concordam nas fendas e discordam nas bordas; a média apaga parte das bordas erradas (precisão +7 pontos), sem aprender nada novo. Segundo, as funções de erro clDice e clCE foram feitas para corrigir ramos falsos, mas o erro principal do nosso modelo é a largura da fenda (Figura 5), então elas não têm o que consertar. O clDice treinado do zero chega a piorar a precisão, porque não leva a largura em conta. Terceiro, treinar com recortes rotulados pelo próprio modelo repete os vícios do modelo: ele marca um pouco além da fenda e ensina isso de novo, então o ganho em revocação é pago em precisão. No conjunto, os testes confirmam o que a Seção 5 mostra: o limite está no gabarito, não na rede.

# 4. Resultados: as fendas e a perda de gelo (2016–2019)

## 4.1 Mapas e contagem de fendas

**Tabela 5 — Contagem anual de fendas (mapas finais, grade de 8 cm).**

| Ano | Manchas marcadas | Fendas com forma bem definida | Área total (mil m²) | Área coberta (%) | Área típica (m²) | Comprimento típico (m) | Altitude típica (m) |
|-------|-------------|---------------|-------------|------------|------------|--------------|--------------|
| 2016 | 11 551 | 10 083 | 59,9 | 1,40 | 1,95 | 6,4 | 132,6 |
| 2017 | 13 297 | 11 526 | 52,3 | 1,22 | 1,41 | 5,3 | 141,0 |
| 2018 | 14 885 | 12 939 | 63,5 | 1,48 | 1,52 | 5,6 | 138,9 |
| 2019 | 17 500 | 14 131 | 50,6 | 1,18 | 0,65 | 3,8 | 129,9 |

*"Típico" = mediana (metade das fendas está acima e metade abaixo). A área coberta muda com o limiar (0,6) e com as condições de cada voo; as comparações mais seguras entre os anos são a contagem por tamanho, a forma e a posição das fendas (Figuras 8 a 11).*

```{=openxml}
<w:p><w:pPr><w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1418" w:right="1418" w:bottom="1418" w:left="1418" w:header="709" w:footer="709" w:gutter="0"/></w:sectPr></w:pPr></w:p>
```

![](../results/figures/relatorio/mapa_fendas_2019_qgis.png){height=15cm}

*Figura 6 — Fendas encontradas pelo modelo em 2019, sobre o ortomosaico de 8 cm, com curvas de nível a cada 50 m e o limite do glaciar. Quadros ampliados (a–d) em resolução original: (a) região da frente, junto ao lago; (b) campo denso no centro; (c) rachaduras em superfície com neve, na parte média; (d) rachaduras finas do lado leste. São 17.500 fendas, cobrindo 1,18% da área. Coordenadas UTM zona 19S.*

```{=openxml}
<w:p><w:pPr><w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/><w:pgMar w:top="851" w:right="851" w:bottom="851" w:left="851" w:header="425" w:footer="425" w:gutter="0"/></w:sectPr></w:pPr></w:p>
```

![](../results/figures/relatorio/mapa_mascaras_4anos_qgis.png){height=15cm}

*Figura 7 — Fendas encontradas sobre os ortomosaicos de 8 cm, de 2016 a 2019. Os quadros ampliados mostram sempre o mesmo trecho do centro do glaciar (158 × 96 m), onde se vê as fendas ficando mais fragmentadas a cada ano. Fendas em vermelho; limite do glaciar em preto.*

```{=openxml}
<w:p><w:pPr><w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/><w:pgMar w:top="851" w:right="851" w:bottom="851" w:left="851" w:header="425" w:footer="425" w:gutter="0"/></w:sectPr></w:pPr></w:p>
```

![](../results/figures/relatorio/fig12_evolucao_anual.png){width=12cm}

*Figura 8 — Série de 2016 a 2019: número de fendas, área total, área coberta e altitude típica. A faixa sombreada é uma margem de incerteza calculada a partir da precisão e da revocação do teste (estimativa cautelosa).*

## 4.2 Fendas mais numerosas e menores

![](../results/figures/relatorio/fig10_morfometria.png){width=15cm}

*Figura 9 — Tamanho das fendas por ano: distribuição da área e do comprimento, curva acumulada da área e largura.*

Entre 2016 e 2019, o número de fendas cresce 40%, enquanto a área total cai 16%. Todo o crescimento vem das **fendas pequenas**: as de menos de 1 m² passam de 2.984 para 6.554, ou de 29,6% para 46,4% do total (teste qui-quadrado, p < 10⁻¹⁵⁰). Ao mesmo tempo, as fendas com mais de 5 m² perdem 745 unidades e cerca de 12,8 mil m². No conjunto de fendas com forma bem definida, a área típica cai 57% (de 2,72 para 1,18 m²), o comprimento cai 41% e a largura 29%, enquanto a proporção entre comprimento e largura fica quase igual (de ~10 para 9). As fendas grandes de 2019 são, em sua maioria, as mesmas de 2016. As pequenas misturam rachaduras novas e pedaços de fendas antigas que se dividiram. Em resumo, **a rede de fendas está se fragmentando**.

## 4.3 Deslocamento, persistência e direção

![](../results/figures/relatorio/fig08_hipsometria.png){width=13cm}

*Figura 10 — Perda de gelo (dH), quantidade de fendas e número de fendas em cada faixa de 50 m de altitude.*

![](../results/figures/relatorio/fig11_rosetas_orientacao.png){width=15cm}

*Figura 11 — Direção das fendas em cada ano (rosetas; o tamanho de cada pétala mostra quantas fendas seguem aquela direção).*

- **As fendas estão subindo no glaciar.** Entre 55% e 63% das fendas de 2019 (conforme o critério usado) não existiam em 2016. Elas ficam mais acima: cerca de 18 m mais alto e 250 m mais longe da frente que as fendas que persistiram (teste de Mann-Whitney, p ≈ 10⁻¹²¹). A altitude típica das fendas cai só 2,7 m, enquanto a superfície baixou cerca de 9,5 m no período. Ou seja, as fendas novas que surgem mais acima compensam o rebaixamento do gelo.
- **O campo de fendas se renova quase por inteiro.** Só 5,5% (sobreposição exata) a 30,3% (com folga de ~1 m) da área de fendas de 2019 já era fenda em 2016. Parte disso vem do próprio movimento do gelo, que carrega as fendas (esse efeito não foi corrigido), e parte vem de rachaduras novas.
- **A direção principal se mantém.** As fendas seguem sobretudo a direção leste-nordeste (65–66°), com um leve giro para 70° e mais variação em 2019 (a queda na concentração das direções não é acaso). Isso é compatível com uma mudança nas forças dentro do gelo enquanto o glaciar recua. A inclinação do terreno onde estão as fendas aumenta (de 20,9° para 22,9°), acompanhando o aumento da inclinação da superfície como um todo.

## 4.4 Perda de gelo e a ligação com as fendas (atividade 3)

**Tabela 7 — Quanto a superfície baixou em cada par de anos, e a conferência em rocha.**

| Par de anos | Queda típica no gelo (m) | Queda típica em toda a área (m) | Diferença em rocha (m) | Quadrados de rocha usados |
|----------------|----------------------|------------------------|--------------------|------------------|
| 2016→17 | −5,31 | −4,50 | +0,04 | 61 |
| 2017→18 | −3,25 | −2,47 | +0,01 | 65 |
| 2018→19 | −3,55 | −2,49 | −0,01 | 67 |

```{=openxml}
<w:p><w:pPr><w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1418" w:right="1418" w:bottom="1418" w:left="1418" w:header="709" w:footer="709" w:gutter="0"/></w:sectPr></w:pPr></w:p>
```

![](../results/figures/relatorio/mapa_dh.png){width=26cm}

*Figura 12 — Quanto a superfície baixou (dH) em cada par de anos (2016–17, 2017–18, 2018–19), numa grade de 2 m. Vermelho = perdeu altura; azul = ganhou altura; a escala vai de −12 a +12 m, com o zero em branco. As faixas azuis e vermelhas alongadas no centro coincidem com o campo de fendas e vêm, em boa parte, do relevo das fendas carregado pelo gelo de um ano para o outro: em 86% dos pontos testados, comparar o mesmo pedaço de gelo em vez do mesmo ponto do chão deixa a diferença mais lisa (Seção 5). Em rocha, que não deveria mudar, a diferença ficou em no máximo 4 cm, o que confirma que os mapas estão bem alinhados.*

```{=openxml}
<w:p><w:pPr><w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/><w:pgMar w:top="851" w:right="851" w:bottom="851" w:left="851" w:header="425" w:footer="425" w:gutter="0"/></w:sectPr></w:pPr></w:p>
```

![](../results/figures/relatorio/mapa_densidade.png){height=15cm}

*Figura 13 — (a–d) Quantidade de fendas em cada quadrado de 100 m, em % da área, de 2016 a 2019. (e) Em quantos dos quatro anos cada quadrado teve fenda (mais de 0,5% da área); os quadrados hachurados tiveram pouca ou nenhuma fenda em todos os anos. (f) Mudança entre 2016 e 2019: azul = menos fendas, vermelho = mais fendas. Só aparecem os quadrados com pelo menos metade da área dentro do glaciar.*

```{=openxml}
<w:p><w:pPr><w:sectPr><w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/><w:pgMar w:top="851" w:right="851" w:bottom="851" w:left="851" w:header="425" w:footer="425" w:gutter="0"/></w:sectPr></w:pPr></w:p>
```

**Tabela 6 — Fendas × perda de gelo em quadrados de 100 m sobre o gelo.** *Em itálico, os valores que podem ser só acaso (p ≥ 0,05).*

| Par de anos | Nº de quadrados | Correlação (ρ) | Correlação descontando a altitude | Diferença: 25% com mais fendas − sem fendas |
|------------------|----------------|---------------------|---------------------|-------------------------|
| 2016→17 | 215 | **−0,58** (p < 0,001) | **−0,39** (p < 0,001) | **−1,97 m** (p < 0,001) |
| 2017→18 | 217 | +0,19 (p = 0,004) | +0,25 (p < 0,001) | *+0,27 m (p = 0,17)* |
| 2018→19 | 216 | *−0,01 (p = 0,91)* | *+0,02 (p = 0,80)* | *−0,11 m (p = 0,86)* |

![](../results/figures/relatorio/fig09_scatter_densidade_dh.png){width=16cm}

*Figura 14 — Quantidade de fendas × perda de gelo nos quadrados de 100 m, para cada par de anos, com a correlação, o valor p e o número de quadrados. Os anos sem relação aparecem no mesmo formato, porque o fato de a ligação aparecer só em um ano também é um resultado.*

No ano de maior perda (2016→17), os quadrados com mais fendas baixaram bem mais: o gelo sem fendas perdeu 4,25 m, e os 25% de quadrados com mais fendas perderam 6,21 m. São **cerca de 2 m a mais de perda ligados às fendas**. O resultado se mantém quando se desconta a altitude e quando se usam quadrados de 100 a 200 m. Nos anos de perda moderada, a ligação some (2018→19) ou aparece fraca e no sentido contrário (2017→18), o que sugere que o clima de cada ano controla esse efeito. Um detalhe importante: quando a análise é feita por faixas de altitude, como é costume, a ligação desaparece (ρ = +0,06). A diferença que importa está de um lado para o outro do glaciar, numa escala de centenas de metros, e não de cima para baixo. Isso é também um resultado de método: a forma de análise mais comum não teria encontrado a ligação. Até onde a nossa busca alcançou, medidas diretas como esta, em números e região por região, eram raras, embora Colgan et al. (2016) já descrevessem os mecanismos.

## 4.5 Testes exploratórios: velocidade, deformação e profundidade

Dois testes inspirados em Duan et al. (2025) mostram até onde os dados atuais chegam.

**(1) Velocidade do gelo.** Tentamos acompanhar as mesmas feições de 2018 para 2019, comparando pedaços de 128 m entre as duas imagens. O alinhamento foi conferido em rocha (deslocamento típico de 0,00 m). Em um ano, porém, **79% da superfície do gelo muda tanto que não dá para reconhecer** as feições, então um mapa completo de velocidade é inviável com um voo por ano. Mesmo assim, os 173 vetores aprovados fazem sentido: 77% apontam para sudoeste/oeste (na direção do lago da frente), e a velocidade típica é de cerca de 91 m por ano. No interior do glaciar, o gelo anda em ritmo parecido perto da frente e mais acima (medianas de 92 e 96 m por ano) e acelera até ~145 m por ano na parte mais alta, a leste. Os pontos quase parados (0 a 20 m por ano) ficam, em sua maioria, a menos de 100 m da borda do glaciar (mediana de 3 m por ano nesses 38 pontos). Podem ser gelo parado junto às margens ou feições da própria borda que não se movem; com um único par de anos não dá para separar os dois casos. A partir desses vetores calculamos em que direção o gelo mais se estica (Nye, 1959). Pela teoria (Harper et al., 1998), as fendas deveriam ficar perpendiculares a essa direção. O ângulo típico medido foi de **73,6°** (74% acima de 60°; o ideal seria 90°). É uma confirmação física, e independente, de que a direção das fendas mapeadas está certa.

![](../results/figures/relatorio/fig15_velocidade.png){width=16cm}

*Figura 15 — Velocidade da superfície de 2018 para 2019. (a) As setas mostram para onde o gelo anda: de leste para oeste, em direção à frente e ao lago. São os 173 pontos que puderam ser acompanhados, coloridos pela velocidade; no resto da superfície (79%) as feições mudaram demais em um ano. (b) Velocidade de cada ponto conforme a distância até a frente: 0 é a frente, a oeste, e a distância cresce subindo o glaciar, para leste. Os círculos vazados são pontos a menos de 100 m da borda do glaciar, quase todos parados. Os pontos do interior andam entre ~85 e ~95 m por ano desde perto da frente e aceleram até ~145 m por ano na parte mais alta. Coordenadas UTM zona 19S.*

**(2) Profundidade das fendas.** Tentamos medir a profundidade das fendas comparando o fundo de cada fenda com a superfície ao redor. O mapa de altura feito a partir das fotos (pixels de ~31 cm) não "entra" nas fendas: a profundidade aparente típica ficou entre 0,14 e 0,16 m (máximo de 0,57 m), e cerca de 90% das fendas não passaram no controle de qualidade. Com LiDAR (sensor a laser) embarcado, Duan et al. (2025) mediram 3,06 ± 3,91 m, cerca de dez vezes mais. Isso mostra, com números, que medir a profundidade das fendas exige LiDAR (trabalho futuro).

```{=openxml}
<w:p><w:r><w:br w:type="page"/></w:r></w:p>
```

# 5. Discussão e limitações

**O modelo chegou ao limite do próprio gabarito.** Para medir o quanto o gabarito é consistente, o anotador marcou de novo, às cegas e mais de dois meses depois, 15 recortes que já tinha marcado. A concordância consigo mesmo foi **F1 = 0,583** e, com folga de 32 cm, **0,785**. O modelo, testado em blocos nunca vistos, tem F1 de 0,661, ou seja, **concorda com o gabarito mais do que o anotador concorda consigo mesmo**. Com folga, a diferença se mantém (0,846 do modelo nos 218 recortes; 0,899 nos mesmos 15 recortes). As diferenças do anotador aparecem tanto em marcar a mais quanto a menos (precisão e revocação ≈ 0,58) e ficam na largura e nas bordas das fendas, que é também o principal "erro" do modelo. A consequência prática é clara: aumentar mais o F1 pixel a pixel seria ajustar o modelo ao ruído das anotações, e não às fendas reais. Por isso paramos de otimizar o modelo e priorizamos a análise do glaciar. Os testes da Seção 3.4 confirmam esse limite: nenhuma mudança no treino superou o controle, e o único ganho repetível, o comitê, age suavizando as bordas. *(Ressalvas: foram 15 recortes e um único anotador. Um teste com mais de um anotador, que é mais forte, está planejado para a publicação do conjunto de dados.)*

**O resultado é bom quando se considera a forma de teste.** Comparar F1 entre estudos sem olhar o teste engana (Tabela 3). Os valores mais altos da literatura vêm de testes na mesma área do treino, com divisão não informada, e o mais alto usa também LiDAR. O único estudo com áreas de teste separadas não informa o F1 da classe fenda. Com a correção de Kattenborn et al. (2022), o nosso resultado equivale à faixa publicada. Na única medida com comparação direta (AUC-ROC), empatamos com os valores publicados na revista *The Cryosphere* (0,93 e 0,91).

**Limitações:**

- *Alinhamento dos mapas de altura:* não foi feito um ajuste fino automático, mas a diferença medida em rocha é de no máximo 4 cm, bem menor que a perda de gelo.
- *Movimento do gelo não corrigido:* "fenda nova" e "fenda que persistiu" são medidas em lugar fixo. Como o gelo anda e carrega as fendas em três anos, a persistência real é maior que a medida (por isso mostramos a faixa de 5,5% a 30,3%). O mesmo vale para a perda de gelo pixel a pixel: um teste em 2018→2019 (`19_teste_deslocamento_dh.py`) comparou, em 128 pontos do interior, a diferença de altura no mesmo ponto do chão com a diferença no mesmo pedaço de gelo, usando a velocidade medida em cada ponto. O sobe-e-desce típico caiu de 2,5 para 1,6 m, e 86% dos pontos ficaram mais lisos. Além disso, o deslocamento que mais alisa o mapa de altura ficou, em mediana, a 5 m da velocidade medida nas fotos. Ou seja, boa parte das faixas de "ganho" e "perda" de pequena escala é o relevo das fendas se deslocando, e não derretimento. Por isso as análises usam medianas em quadrados de 100 m, onde esse efeito se cancela em grande parte.
- *Gabarito de 2017 a 2019:* carrega o estilo do modelo antigo (Seção 2.2). As notas desses anos são indicativas; a avaliação principal é a de 2016 em blocos nunca vistos.
- *Limiar:* a área coberta por fendas em cada ano muda com o limiar (0,6). As conclusões principais (fragmentação, subida das fendas e ligação com a perda de gelo) não mudam: elas se repetiram em três versões sucessivas do modelo.
- *Fendas muito pequenas:* manchas com menos de ~0,5 m² não são detectadas, mas respondem por só 8% da área de fendas perdidas. Uma medida de distância entre as bordas (HD95) mostra que, nas fendas encontradas, o erro de borda é de 0,8 a 1,7 m; os erros maiores vêm de fendas finas que o modelo não encontrou por inteiro, e não de bordas mal desenhadas.
- *Quadrados vizinhos são parecidos:* por isso os valores de p são otimistas, e sempre mostramos o tamanho do efeito em metros junto com o p.
- *Série curta:* quatro anos não bastam para falar em tendência; as séries anuais são descritivas.

# 6. Conclusões

1. Uma U-Net com seis camadas de entrada, treinada com um único ano de fotos, marca fendas em fotos de drone de 8 cm com **F1 de 0,661 ± 0,018** em regiões nunca vistas (0,663 com outra divisão dos blocos). Funciona nos anos seguintes sem novo treino (0,66 a 0,71; 0,69 a 0,71 com o comitê) e chega a F1 de 0,85 com folga de 32 cm. Considerando a forma de teste, o resultado é competitivo com a literatura, e os recortes anotados formam um dos primeiros conjuntos de fendas em fotos de drone, o primeiro testado em regiões separadas e com quatro anos de série.
2. **Revisar o gabarito foi o passo mais importante.** Quase metade dos recortes "sem fenda" não conferidos tinha fendas; corrigi-los valeu +5 pontos de precisão, e a revisão final de 18 recortes somou +0,6 ponto de F1 na medição, mais que qualquer mudança de modelo testada. Um **comitê de dez modelos** leva o F1 dos outros anos a 0,70 (precisão 0,70). Outras formas de calcular o erro, limpezas da resposta e o treino com fotos não anotadas não superaram o ruído das anotações.
3. As fendas do Glaciar Schiaparelli **ficam mais fragmentadas e sobem no glaciar** entre 2016 e 2019: são 40% mais numerosas por causa das pequenas, ocupam 16% menos área, renovam-se quase por inteiro em três anos, e as novas surgem cerca de 250 m mais longe da frente.
4. **A ligação entre fendas e perda de gelo existe, mas aparece em episódios.** No ano de maior derretimento, o gelo com muitas fendas perdeu cerca de 2 m a mais que o gelo sem fendas (correlação descontando a altitude de −0,39). A ligação não aparece nos anos de perda moderada nem na análise por faixas de altitude. É uma medida direta, na escala de 100 a 200 m, de que as fendas ajudam a acelerar a perda de gelo.

**Próximos passos:** publicar o conjunto de dados anotados; montar um teste independente com as fotos de 2020; usar o comitê de dez modelos nos mapas finais; fazer um teste de concordância entre dois ou mais anotadores e revisar os 22 recortes ainda em dúvida; medir a velocidade do gelo com voos próximos no tempo (dentro da mesma campanha) ou acompanhando à mão as fendas que persistem, já que 79% da superfície muda demais em um ano; medir a profundidade das fendas com LiDAR, já que as fotos só registram cerca de 0,15 m; e estender o método a lagos e canais na superfície do gelo.

# 7. Atividades realizadas × cronograma

| Atividade do plano | Situação |
|----------------------------------------------------------------------|------------------------------|
| 1. Treinamento no Agisoft Metashape / dados processados | Concluída |
| 2. Processamento dos levantamentos de drone (recortes 2016–2020, grade comum de 8 cm) | Concluída |
| 3. Estimativa da perda de gelo na superfície (dH e ligação com as fendas) | **Concluída** |
| 4. Mapeamento com aprendizado de máquina (fendas 2016–2019) | **Concluída** (lagos e canais ficam como extensão futura) |

Principais etapas: troca do SAM pela U-Net (abril de 2026); modelo com seis camadas e variações de cor no treino (maio); grade comum de 8 cm e primeiras análises do glaciar (junho); validação cruzada espacial (junho); revisão do gabarito, correção da montagem dos mapas e mapas finais (agosto); revisão final das anotações, nova validação com duas divisões, comitê de modelos e testes de funções de erro e de treino com fotos não anotadas (setembro).

# Referências

- Acebes, C. et al. (2024). The centerline-cross entropy loss for vessel-like structure segmentation: better topology consistency at a lower cost. *MICCAI*.
- Baumhoer, C. et al. (2025). Automated crevasse mapping for Alpine glaciers. *Int. J. Appl. Earth Obs. Geoinf.* 139:104495.
- Benz, C.; Rodehorst, V. (2024). OmniCrack30k: a benchmark for crack segmentation. *CVPR Workshops*.
- Colgan, W. et al. (2016). Glacier crevasses: observations, models, and mass balance implications. *Reviews of Geophysics* 54(1):119–161.
- Duan, Y. et al. (2025). Automated high-resolution 3D crevasse extraction: C-TransUNet. *Int. J. Appl. Earth Obs. Geoinf.* 144:104881.
- Harper, J. T.; Humphrey, N. F.; Pfeffer, W. T. (1998). Crevasse patterns and the strain-rate tensor: a high-resolution comparison. *Journal of Glaciology* 44(146):68–76.
- Hugonnet, R. et al. (2021). Accelerated global glacier mass loss in the early twenty-first century. *Nature* 592.
- Kattenborn, T. et al. (2022). Spatially autocorrelated training and validation samples inflate performance assessment of CNNs. *ISPRS Open J. Photogramm. Remote Sens.* 5:100018.
- Kneib, M. et al. (2021). Mapping ice cliffs on debris-covered glaciers. *Remote Sens. Environ.* 253:112201.
- Mnih, V.; Hinton, G. (2010). Learning to detect roads in high-resolution aerial images. *ECCV*.
- Nye, J. F. (1959). A method of determining the strain-rate tensor at the surface of a glacier. *Journal of Glaciology* 3(25):409–419.
- Ploton, P. et al. (2020). Spatial validation reveals poor predictive performance of large-scale ecological mapping models. *Nature Communications* 11.
- Roberts, D. R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography* 40:913–929.
- Shit, S. et al. (2021). clDice — a novel topology-preserving loss for tubular structure segmentation. *CVPR*.
- Surawy-Stepney, T. et al. (2023). Mapping Antarctic crevasses using deep learning. *The Cryosphere* 17:4421–4445.
- Wallace, S. et al. (2025). Exploring Segment Anything foundation models for out-of-domain crevasse drone image segmentation. *NLDL, PMLR* 265:255–268.
- Wallace, S. et al. (2026). CrevasseSeg: a label-efficient UAV crevasse segmentation framework. *arXiv* 2608.15790 (preprint).
- Wiedemann, C. et al. (1998). Empirical evaluation of automatically extracted road axes. *CVPR Workshops*.
- Zhao, J. et al. (2022). Detection of surface crevasses over Antarctic ice shelves. *Remote Sensing* 14(3):487.
- Zheng, L. et al. (2024). CREVNet. *IEEE Geosci. Remote Sens. Lett.*, doi:10.1109/LGRS.2024.3407860.
