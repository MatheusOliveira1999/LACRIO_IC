# Guia de Anotação com QGIS - Glaciar Schiaparelli

Este guia explica como usar o QGIS para anotar manualmente lagos, fendas e canais supraglaciais nos tiles do projeto LACRIO IC.

---

## 📋 Pré-requisitos

- QGIS 3.x instalado (recomendado: 3.28+)
- Tiles gerados em `/tiles/{ano}/` (PNG 512x512)

---

## 🚀 Passo a Passo

### 1. Criar Projeto QGIS

1. Abra o QGIS
2. `Projeto → Novo`
3. Salve como `anotacoes_schiaparelli.qgz` na pasta do projeto

### 2. Carregar Tiles como Raster

1. `Camada → Adicionar Camada → Adicionar Camada Raster`
2. Navegue até `/home/matheus/Documents/GitHub/LACRIO IC/tiles/2016/`
3. Selecione um tile para começar (ex: `tile_000000.png`)

> **Dica:** Para anotar vários tiles, carregue-os em sequência conforme avança.

### 3. Criar Camadas de Anotação (Vetores)

#### Para cada tipo de feição, crie uma camada separada:

**a) Lagos (lakes)**
1. `Camada → Criar Camada → Nova Camada GeoPackage`
   - Nome: `lakes_annotations.gpkg`
   - Tabela: `lakes`
   - Geometria: **Polígono**
   - CRS: Deixe sem CRS (dados não georreferenciados)
2. Adicione campos:
   - `tile_id` (Integer)
   - `year` (Integer)
   - `notes` (Text, opcional)

**b) Fendas (crevasses)**
- Repita os passos acima com:
  - Nome: `crevasses_annotations.gpkg`
  - Tabela: `crevasses`
  - Geometria: **Polígono** ou **Linha** (para fendas lineares)

**c) Canais (channels)**
- Repita os passos com:
  - Nome: `channels_annotations.gpkg`
  - Tabela: `channels`
  - Geometria: **Linha**

### 4. Configurar Simbologia

1. Clique direito na camada → `Propriedades → Simbologia`
2. Configure cores distintas:

| Feição | Cor | Transparência |
|--------|-----|---------------|
| Lakes | Azul (#0000FF) | 50% |
| Crevasses | Vermelho (#FF0000) | 50% |
| Channels | Ciano (#00FFFF) | 50% |

### 5. Anotar Feições

#### Ativar Edição
1. Clique direito na camada → `Alternar Edição` (ícone lápis)

#### Desenhar Polígonos (Lagos/Fendas)
1. Selecione ferramenta `Adicionar Feição Poligonal` (ícone polígono)
2. Clique nos vértices ao redor da feição
3. Clique direito para finalizar
4. Preencha atributos:
   - `tile_id`: ID do tile (ex: 500)
   - `year`: 2016

#### Desenhar Linhas (Canais/Fendas)
1. Selecione `Adicionar Feição Linear`
2. Trace ao longo da feição
3. Clique direito para finalizar

#### Salvar
- `Camada → Salvar Edições da Camada` ou Ctrl+S

### 6. Navegar Entre Tiles

#### Opção A: Carregar um por um
1. Remova tile atual: Clique direito → `Remover`
2. Adicione próximo tile

#### Opção B: Usar Catálogo Virtual (Recomendado)
1. `Raster → Miscelânea → Construir Catálogo Raster Virtual`
2. Adicione todos os tiles de um ano
3. Navegue pelo mosaico virtual

### 7. Exportar Anotações como Máscaras

Após terminar, converta as anotações vetoriais em máscaras raster:

```bash
# Para cada camada de anotação
cd /home/matheus/Documents/GitHub/LACRIO\ IC

# Lagos
gdal_rasterize -burn 255 -ts 512 512 \
  -ot Byte -of PNG \
  lakes_annotations.gpkg \
  masks/2016/annotations/tile_XXXXXX_lakes.png

# Ou use o script de conversão (próxima seção)
```

---

## 📂 Estrutura de Arquivos

```
LACRIO IC/
├── qgis/
│   ├── anotacoes_schiaparelli.qgz   # Projeto QGIS
│   ├── lakes_annotations.gpkg       # Polígonos de lagos
│   ├── crevasses_annotations.gpkg   # Polígonos de fendas
│   └── channels_annotations.gpkg    # Linhas de canais
└── masks/
    └── 2016/
        └── annotations/             # Máscaras convertidas
```

---

## 🔄 Script de Conversão: Vetor → Máscara

Crie o arquivo `convert_qgis_to_masks.py` para automatizar a conversão:

```python
# Uso: python convert_qgis_to_masks.py --year 2016 --feature lakes
```

---

## ✅ Checklist de Anotação

| Item | Meta | Feito |
|------|------|-------|
| Lagos anotados | 20+ tiles | [ ] |
| Fendas anotadas | 15+ tiles | [ ] |
| Canais anotados | 15+ tiles | [ ] |
| Exportação máscaras | Todos | [ ] |

---

## 💡 Dicas

1. **Zoom**: Use a roda do mouse para ampliar detalhes
2. **Pan**: Barra de espaço + arrastar
3. **Snapping**: Ative em `Projeto → Opções de Sketching` para precisão
4. **Atalhos úteis**:
   - `E`: Alternar edição
   - `Ctrl+Z`: Desfazer
   - `Del`: Excluir feição selecionada

---

## 🎯 Critérios de Anotação

### Lagos
- Áreas de água visíveis (azul/turquesa)
- Incluir poças maiores que ~10 pixels

### Fendas
- Fraturas lineares escuras no gelo
- Podem ser desenhadas como polígonos ou linhas

### Canais
- Rede de drenagem supraglacial
- Desenhar como linhas seguindo o fluxo

---

*Guia criado para o projeto LACRIO IC - Janeiro 2026*
