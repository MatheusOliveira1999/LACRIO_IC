# 🧊 LACRIO IC - Análise de Geleiras com SAM

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Projeto de Iniciação Científica** para análise de mudanças na superfície de geleiras utilizando dados de VANT (Veículos Aéreos Não Tripulados) e técnicas de Machine Learning.

> **Orientador:** Prof. Dr. Jorge Arigony Neto  
> **Laboratório:** LaCrio - Laboratório de Monitoramento da Criosfera (FURG)  
> **Bolsa:** CNPq - Vigência Set/2025 – Ago/2026

---

## 📋 Objetivos

1. **Mapeamento de feições supraglaciais** usando SAM (Segment Anything Model)
   - Lagos supraglaciais
   - Fendas (crevasses)
   - Canais de degelo

2. **Estimativa de ablação superficial** via análise de DEMs multitemporais

3. **Análise da rede hidrológica** e sua relação com o balanço de massa glacial

---

## 🗺️ Área de Estudo

**Glaciar Schiaparelli** - Cordilheira Darwin, Terra do Fogo, Chile

| Dados | Período | Resolução |
|-------|---------|-----------|
| Mosaicos RGB | 2016-2020 | 5.4 cm/pixel |
| DEMs | 2016-2022 | 22 cm/pixel |

---

## 🚀 Instalação

```bash
# Clonar repositório
git clone https://github.com/MatheusOliveira1999/LACRIO_IC.git
cd LACRIO_IC

# Criar ambiente conda
conda create -n sam_glaciar python=3.10 -y
conda activate sam_glaciar

# Instalar PyTorch com CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Instalar dependências
pip install -r requirements.txt

# Baixar modelo SAM (375 MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## 📁 Estrutura do Projeto

```
LACRIO_IC/
├── config.py                  # Configurações centralizadas
├── 01_create_tiles.py         # Gerar tiles 512x512
├── 02_sam_interactive.py      # Anotação interativa SAM
├── 03_finetune_sam.py         # Fine-tuning do modelo
├── 04_inference.py            # Inferência em larga escala
├── 05_reconstruct_mosaic.py   # Reconstruir mosaicos
├── 06_dem_analysis.py         # Análise de DEMs/ablação
├── mosaicos_DEMs_Schiaparelli/  # Dados fonte (não versionados)
├── tiles/                     # Tiles gerados
├── masks/                     # Máscaras de segmentação
├── models/                    # Modelos treinados
└── results/                   # Resultados finais
```

---

## 🔧 Uso

### 1. Gerar Tiles
```bash
# Ver informações dos mosaicos
python 01_create_tiles.py --info

# Processar todos os anos
python 01_create_tiles.py

# Processar ano específico
python 01_create_tiles.py --year 2019
```

### 2. Anotação Interativa
```bash
python 02_sam_interactive.py
# Clique esquerdo: ponto positivo
# Clique direito: ponto negativo
# 's': salvar máscara
# 'n'/'p': próximo/anterior
```

### 3. Fine-tuning e Inferência
```bash
python 03_finetune_sam.py
python 04_inference.py
python 05_reconstruct_mosaic.py
```

---

## 📊 Resultados Esperados

| Feição | F1-Score | IoU |
|--------|----------|-----|
| Lagos supraglaciais | 85-90% | 75-85% |
| Fendas | 80-85% | 70-80% |
| Canais de degelo | 75-85% | 65-75% |

---

## 📚 Referências

- Kirillov, A. et al. (2023). **Segment Anything**. ICCV 2023.
- Chai, M. et al. (2025). **Potential of SAM for supraglacial lakes**. Int. J. Digital Earth.

---

## 📄 Licença

Este projeto está sob a licença MIT.

---

## 👤 Contato

**Matheus Oliveira**  
Graduação em Oceanologia - FURG  
GitHub: [@MatheusOliveira1999](https://github.com/MatheusOliveira1999)
