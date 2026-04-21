# Guia de Anotação — Lagos Supraglaciais

**Projeto**: LACRIO IC — Extração de Feições Supraglaciais (Glaciar Schiaparelli)

Este guia define, de forma objetiva, o que deve (e o que **não** deve) ser
anotado como **lago supraglacial** nas máscaras de treino do modelo U-Net.

A consistência entre tiles é o fator mais importante para a qualidade do
modelo. Quando em dúvida, **não anote** — um falso negativo (perder um lago)
é menos prejudicial que um falso positivo (ensinar que rocha/sombra é lago).

---

## 1. Definição

**Lago supraglacial** é um corpo d'água **parada** (*ponded*) formado pelo
derretimento superficial do gelo, que se acumula em depressões na parte de
cima do glaciar.

### Características visuais

| Propriedade | Descrição |
|-------------|-----------|
| **Cor**      | Azul claro → azul escuro → quase preto (profundidade define a tonalidade) |
| **Borda**    | Fechada, contínua, contorno de bacia/poça |
| **Forma**    | Arredondada, oval ou irregular **sem direção preferencial** |
| **Textura**  | Interior liso, uniforme, sem granulosidade |
| **Posição**  | Em áreas relativamente planas do gelo, não em declives acentuados |

> **Regra mental**: *"Se eu esvaziasse essa água, sobraria uma bacia/depressão
> arredondada?"* — se sim, é lago. Se sobraria uma trincheira linear, não é.

---

## 2. O que anotar (✅)

- Poças com **borda fechada** e **forma arredondada/oval/irregular**
- Cores azul turquesa, azul escuro **e também pretas** — lagos profundos
  (>2m) absorvem quase toda a luz e ficam quase pretos. **Preto liso + borda
  fechada = lago.**
- Lagos parcialmente cobertos por gelo flutuante (anote a extensão total)
- Lagos grandes que se estendem por vários tiles (anote a parte dentro do
  tile)
- Duas poças conectadas por um trecho curto de **água parada** (mesma cor e
  textura em tudo): anote como um único lago

## 3. O que NÃO anotar (❌)

### 3.1 Canais/riachos supraglaciais (*supraglacial streams*)

- Lineares, sinuosos, com **direção preferencial** (seguem a inclinação)
- Estreitos, longos
- Mesmo que azulados/escuros — é **água em movimento**, não lago

### 3.2 Fendas (*crevasses*)

- Padrão de **linhas paralelas** (fracturas por tensão)
- Pretas no interior por causa de **sombra profunda**, não água
- Bordas retas, não arredondadas
- Direção preferencial alinhada ao campo de tensão do gelo

### 3.3 Rede de drenagem entre fendas (*crevasse-controlled drainage*)

- Áreas escuras **alongadas na direção das fendas**
- Parecem "poças entre fendas" mas são depressões preenchidas com mistura
  de água/neve derretida (*slush*) + sombra
- Se você anotar isso como lago, o modelo vai errar em toda zona de fendas

### 3.4 Rocha / detrito (*debris*)

- **Textura granulosa**, "pedregulhenta", não lisa
- Cor geralmente marrom/cinza, mas pode ser preto
- Sem brilho especular

### 3.5 Sombra topográfica

- Preta mas segue o relevo
- Borda muito nítida de um lado, gradiente do outro
- Forma coerente com feições topográficas vizinhas (ex.: sombra de serra)

### 3.6 Neve/gelo sujo

- Cinza escuro mas com textura (cristais)
- Não é uniformemente escuro, tem "grãos" visíveis

---

## 4. Casos ambíguos frequentes

### "Parece lago mas tem canal saindo"

- O **lago em si** (área represada) → **anote**
- O **canal de saída** (*outflow*, trecho estreito que drena para fora)
  → **NÃO anote**
- Corte a anotação onde o lago se estreita em canal

### "Degelo escuro sem borda clara"

- Se não consegue distinguir onde termina → **não anote**
- Melhor errar pra menos

### "Duas poças conectadas por canal fino"

- Se o canal é **claramente fluindo** (linear, mais claro, direção definida)
  → anote as duas poças separadas, **ignore o canal**
- Se a conexão é **larga e represada** (mesma cor, água parada) → anote
  tudo junto como um lago só

### "Lago profundo, preto, bordas difusas"

- Se a **forma geral é arredondada** e o interior é **liso** → anote
- Se a borda "vaza" em várias direções tipo árvore → é rede de drenagem,
  **não anote**

---

## 5. Checklist rápido (3 perguntas)

Para cada feição escura, pergunte-se:

1. A borda é **fechada e arredondada** (não linear)?
2. O interior é **liso e uniforme** (não texturizado)?
3. A feição está em uma área **plana** do gelo (não em declive acentuado)?

**Sim para as 3** → é lago, anote.
**Não para qualquer uma** → provavelmente não é lago, deixe em branco.

---

## 6. Precisão das bordas

**Não precisa ser perfeito pixel a pixel.** O modelo aprende o padrão visual
(cor + textura + forma) e generaliza bordas a partir dos exemplos.

- ✅ Erros de 5–10 pixels nas bordas são inofensivos
- ✅ Cobrir bem o interior é o que importa
- ❌ Não deixar grandes buracos não anotados dentro do lago
- ❌ Não incluir grande área de gelo ao redor

---

## 7. Consistência > Cobertura

Se você anotar um tipo de feição em alguns tiles mas não em outros, o modelo
aprende ruído. Escolha uma política e siga em todos os tiles:

- **Política recomendada**: anotar apenas **lagos clássicos** (poças
  represadas, borda fechada, forma arredondada). Ignorar redes de drenagem,
  slush swamps e água em fendas.

Se quiser mudar de política no meio do projeto, **re-revise todos os tiles
anotados antes** com o novo critério — use `python annotate.py --review`.

---

## 8. Workflow sugerido

1. **Anotar tiles novos**: `python annotate.py --year 2017`
2. **Revisar anotações antigas**: `python annotate.py --review --year 2016`
3. **Quando em dúvida**: abrir o tile em tamanho maior (Ctrl+Scroll para
   zoom até 64x) e comparar com tiles vizinhos já anotados

---

## Referências

- Pope, A. et al. (2016). *Estimating supraglacial lake depth in West
  Greenland using Landsat 8 and comparison with other multispectral
  methods*. The Cryosphere, 10(1), 15–27.
- Williamson, A. G. et al. (2018). *A fully automated supraglacial lake
  area and volume tracking ("FAST") algorithm*. Remote Sensing of
  Environment, 196, 113–133.
- 2023 SIGSPATIAL GIS Cup — dataset de treino com definições vetoriais
  de lagos supraglaciais, canais e fendas (referência para a nossa
  taxonomia).
