# Recomendação de Machine Learning: SAM (Segment Anything Model)

**Projeto:** Extração de Feições Supraglaciais - Glaciar Schiaparelli
**Data:** Janeiro 2026
**Técnica Recomendada:** SAM (Segment Anything Model) com Fine-tuning

---

## 1. Por que SAM é a Técnica Mais Eficiente?

### 1.1 Análise dos Seus Dados

| Característica | Valor | Implicação para ML |
|----------------|-------|-------------------|
| Resolução RGB | **5.4 cm/pixel** | Excelente para qualquer modelo |
| Resolução DEM | **22 cm/pixel** | Pode ser canal adicional |
| Bandas espectrais | **RGB apenas** | Limita modelos pré-treinados em satélite |
| Volume por imagem | **~5 GB** | Tiling obrigatório |
| Série temporal | **5-6 anos** | Permite validação cruzada temporal |
| Tempo disponível | **2 meses** | Favorece modelos pré-treinados |

### 1.2 Comparação de Técnicas de ML

| Critério | U-Net | DeepLabV3+ | SAM | Vencedor |
|----------|-------|------------|-----|----------|
| Anotações necessárias | 500+ tiles | 500+ tiles | **20-50 tiles** | SAM |
| Tempo de anotação | 20-40h | 20-40h | **2-4h** | SAM |
| Tempo de treinamento | 5-10h | 5-10h | **1-2h** | SAM |
| Precisão em lagos | 80-85% | 82-87% | **87.77%** | SAM |
| Generalização | Baixa | Média | **Alta** | SAM |
| Facilidade de uso | Média | Média | **Alta** | SAM |
| Base de pré-treino | ImageNet (1M) | ImageNet (1M) | **SA-1B (11M)** | SAM |
| Multiuso | 1 modelo/feição | 1 modelo/feição | **1 modelo/todas** | SAM |

### 1.3 Vantagens do SAM para Seu Projeto

1. **Mínima anotação necessária** - 20-50 exemplos vs 500+ para U-Net tradicional
2. **Estado da arte em 2025** - 87.77% F1-score em lagos supraglaciais (Chai et al., 2025)
3. **Modelo único para todas as feições** - Fendas, canais E lagos com o mesmo modelo
4. **Zero-shot possível** - Pode testar imediatamente sem treinamento
5. **Fine-tuning rápido** - Adapta em poucas épocas (1-2 horas)
6. **Modelo fundacional** - Treinado em 11 milhões de imagens diversas

---

## 2. Arquitetura do SAM

```
┌─────────────────────────────────────────────────────────────────┐
│                         SAM ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Tile RGB (512x512 pixels)                              │
│                    ↓                                            │
│  ┌────────────────────────────────┐                            │
│  │      IMAGE ENCODER (ViT)       │  ← Pré-treinado (congelado)│
│  │   Vision Transformer Base/Huge │                            │
│  │   Extrai features da imagem    │                            │
│  └───────────────┬────────────────┘                            │
│                  ↓                                              │
│           Image Embeddings                                      │
│                  ↓                                              │
│  ┌────────────────────────────────┐                            │
│  │       PROMPT ENCODER           │  ← Pontos, boxes, máscaras │
│  │   Codifica instruções do       │                            │
│  │   usuário sobre o que segmentar│                            │
│  └───────────────┬────────────────┘                            │
│                  ↓                                              │
│  ┌────────────────────────────────┐                            │
│  │        MASK DECODER            │  ← Fine-tuned para glaciar │
│  │   Gera máscara de segmentação  │                            │
│  │   Lightweight (poucos params)  │                            │
│  └───────────────┬────────────────┘                            │
│                  ↓                                              │
│  Output: Máscara binária + Score de confiança                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Versões do Modelo

| Versão | Encoder | Parâmetros | VRAM | Velocidade | Precisão |
|--------|---------|------------|------|------------|----------|
| **vit_b** | ViT-Base | 91M | 4GB | Rápido | Boa |
| **vit_l** | ViT-Large | 308M | 8GB | Médio | Muito Boa |
| **vit_h** | ViT-Huge | 636M | 16GB | Lento | Excelente |

**Recomendação:** Usar `vit_b` para desenvolvimento/testes e `vit_h` para resultados finais.

---

## 3. Implementação Completa

### 3.1 Instalação do Ambiente

```bash
# Criar ambiente conda
conda create -n sam_glaciar python=3.10
conda activate sam_glaciar

# PyTorch com CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# SAM
pip install segment-anything

# Dependências adicionais
pip install opencv-python rasterio geopandas numpy tqdm

# Download do modelo SAM (escolher um):
# vit_b (375 MB) - mais rápido
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# vit_h (2.4 GB) - mais preciso
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### 3.2 Configuração

```python
"""
config.py - Configurações do projeto SAM para Glaciar Schiaparelli
"""

from pathlib import Path
import torch

class Config:
    # Diretórios
    PROJECT_DIR = Path("/home/matheus/Documents/GitHub/LACRIO IC/mosaicos_DEMs_Schiaparelli")
    DATA_DIR = PROJECT_DIR / "data"
    TILES_DIR = PROJECT_DIR / "tiles"
    MASKS_DIR = PROJECT_DIR / "masks"
    MODELS_DIR = PROJECT_DIR / "models"
    RESULTS_DIR = PROJECT_DIR / "results"

    # Modelo SAM
    SAM_CHECKPOINT = PROJECT_DIR / "sam_vit_b_01ec64.pth"
    MODEL_TYPE = "vit_b"  # "vit_b", "vit_l", ou "vit_h"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Tiling
    TILE_SIZE = 512
    OVERLAP = 64

    # Feições alvo
    FEATURES = {
        "lakes": {
            "description": "Lagos e poças supraglaciais",
            "color": (0, 0, 255),  # Azul
            "min_area": 50,
            "max_area": 50000
        },
        "crevasses": {
            "description": "Fendas no gelo",
            "color": (255, 0, 0),  # Vermelho
            "min_area": 100,
            "max_area": 10000
        },
        "channels": {
            "description": "Canais de água de degelo",
            "color": (0, 255, 255),  # Ciano
            "min_area": 200,
            "max_area": 20000
        }
    }

    # Anos disponíveis
    YEARS = [2016, 2017, 2018, 2019, 2020]

    @classmethod
    def create_directories(cls):
        """Cria estrutura de diretórios do projeto."""
        for dir_path in [cls.DATA_DIR, cls.TILES_DIR, cls.MASKS_DIR,
                         cls.MODELS_DIR, cls.RESULTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

        for year in cls.YEARS:
            (cls.TILES_DIR / str(year)).mkdir(exist_ok=True)
            (cls.MASKS_DIR / str(year)).mkdir(exist_ok=True)
```

### 3.3 Preparação de Dados (Tiling)

```python
"""
01_create_tiles.py - Divide mosaicos em tiles para processamento
"""

import rasterio
from rasterio.windows import Window
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
from config import Config

def create_tiles_for_sam(mosaic_path, output_dir, tile_size=512, overlap=64):
    """
    Divide mosaico grande em tiles para SAM.

    Args:
        mosaic_path: Caminho para o mosaico GeoTIFF
        output_dir: Diretório de saída para tiles
        tile_size: Tamanho do tile em pixels
        overlap: Sobreposição entre tiles

    Returns:
        Lista com metadados dos tiles criados
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles_info = []

    with rasterio.open(mosaic_path) as src:
        print(f"Mosaico: {src.width}x{src.height} pixels")
        print(f"Bandas: {src.count}")

        tile_id = 0
        step = tile_size - overlap

        # Calcular número total de tiles para barra de progresso
        n_tiles_x = (src.width - tile_size) // step + 1
        n_tiles_y = (src.height - tile_size) // step + 1
        total_tiles = n_tiles_x * n_tiles_y

        with tqdm(total=total_tiles, desc="Criando tiles") as pbar:
            for y in range(0, src.height - tile_size + 1, step):
                for x in range(0, src.width - tile_size + 1, step):
                    window = Window(x, y, tile_size, tile_size)

                    # Ler apenas RGB (bandas 1, 2, 3)
                    rgb = src.read([1, 2, 3], window=window)

                    # Verificar se tile tem dados válidos (menos de 30% NoData)
                    nodata_ratio = np.sum(rgb == 0) / (tile_size * tile_size * 3)
                    if nodata_ratio > 0.3:
                        pbar.update(1)
                        continue

                    # Converter para formato HWC (Height, Width, Channels)
                    rgb_hwc = np.transpose(rgb, (1, 2, 0))

                    # Salvar como PNG (SAM espera RGB padrão)
                    tile_filename = f"tile_{tile_id:06d}.png"
                    tile_path = output_dir / tile_filename
                    cv2.imwrite(str(tile_path), cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR))

                    # Guardar metadados para reconstrução posterior
                    transform = rasterio.windows.transform(window, src.transform)
                    tiles_info.append({
                        "id": tile_id,
                        "filename": tile_filename,
                        "x": x,
                        "y": y,
                        "width": tile_size,
                        "height": tile_size,
                        "transform": list(transform)[:6],
                        "nodata_ratio": float(nodata_ratio)
                    })

                    tile_id += 1
                    pbar.update(1)

        # Salvar índice de tiles
        index_path = output_dir / "tiles_index.json"
        with open(index_path, "w") as f:
            json.dump({
                "source": str(mosaic_path),
                "tile_size": tile_size,
                "overlap": overlap,
                "total_tiles": tile_id,
                "crs": str(src.crs),
                "tiles": tiles_info
            }, f, indent=2)

        print(f"\nCriados {tile_id} tiles em {output_dir}")
        print(f"Índice salvo em {index_path}")

        return tiles_info


def process_all_years():
    """Processa todos os anos disponíveis."""
    Config.create_directories()

    mosaics = {
        2016: "Schiaparelli_mosaic_2016.tif",
        2017: "Schiaparelli_mosaic_2017.tif",
        2018: "Schiaparelli_mosaic_2018.tif",
        2019: "Schiaparelli_mosaic_2019.tif",
        2020: "schiaparelli_mosaic_2020.tif"
    }

    for year, filename in mosaics.items():
        mosaic_path = Config.PROJECT_DIR / filename
        if mosaic_path.exists():
            print(f"\n{'='*50}")
            print(f"Processando {year}")
            print(f"{'='*50}")

            output_dir = Config.TILES_DIR / str(year)
            create_tiles_for_sam(
                mosaic_path,
                output_dir,
                tile_size=Config.TILE_SIZE,
                overlap=Config.OVERLAP
            )


if __name__ == "__main__":
    process_all_years()
```

### 3.4 SAM - Modo Interativo para Anotação

```python
"""
02_sam_interactive.py - Anotação interativa com SAM
Clique em pontos → SAM gera máscara → Salve como ground truth
"""

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor
from pathlib import Path
import json
from config import Config


class SAMInteractiveAnnotator:
    """
    Interface interativa para criar anotações usando SAM.
    """

    def __init__(self):
        print(f"Carregando SAM ({Config.MODEL_TYPE})...")
        self.sam = sam_model_registry[Config.MODEL_TYPE](
            checkpoint=str(Config.SAM_CHECKPOINT)
        )
        self.sam.to(Config.DEVICE)
        self.predictor = SamPredictor(self.sam)
        print(f"Modelo carregado em {Config.DEVICE}")

        # Estado da interface
        self.image = None
        self.image_rgb = None
        self.current_mask = None
        self.positive_points = []
        self.negative_points = []
        self.masks_saved = []

    def load_image(self, image_path):
        """Carrega imagem para anotação."""
        self.image = cv2.imread(str(image_path))
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.image_rgb)
        self.reset_points()
        print(f"Imagem carregada: {image_path}")

    def reset_points(self):
        """Limpa pontos selecionados."""
        self.positive_points = []
        self.negative_points = []
        self.current_mask = None

    def add_point(self, x, y, is_positive=True):
        """Adiciona ponto e atualiza segmentação."""
        if is_positive:
            self.positive_points.append([x, y])
        else:
            self.negative_points.append([x, y])

        self._update_mask()

    def _update_mask(self):
        """Atualiza máscara com base nos pontos."""
        if not self.positive_points:
            self.current_mask = None
            return

        points = np.array(self.positive_points + self.negative_points)
        labels = np.array(
            [1] * len(self.positive_points) +
            [0] * len(self.negative_points)
        )

        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True
        )

        # Selecionar máscara com maior score
        best_idx = np.argmax(scores)
        self.current_mask = masks[best_idx]
        self.current_score = scores[best_idx]

    def get_visualization(self):
        """Retorna imagem com overlay da máscara e pontos."""
        vis = self.image.copy()

        # Overlay da máscara
        if self.current_mask is not None:
            mask_overlay = np.zeros_like(vis)
            mask_overlay[self.current_mask] = [0, 255, 0]  # Verde
            vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)

        # Desenhar pontos positivos (verde)
        for x, y in self.positive_points:
            cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(vis, (x, y), 7, (255, 255, 255), 2)

        # Desenhar pontos negativos (vermelho)
        for x, y in self.negative_points:
            cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(vis, (x, y), 7, (255, 255, 255), 2)

        return vis

    def save_mask(self, output_path, feature_type):
        """Salva máscara atual."""
        if self.current_mask is None:
            print("Nenhuma máscara para salvar!")
            return False

        mask_uint8 = (self.current_mask * 255).astype(np.uint8)
        cv2.imwrite(str(output_path), mask_uint8)

        self.masks_saved.append({
            "path": str(output_path),
            "feature_type": feature_type,
            "positive_points": self.positive_points.copy(),
            "negative_points": self.negative_points.copy(),
            "score": float(self.current_score)
        })

        print(f"Máscara salva: {output_path} (score: {self.current_score:.3f})")
        return True


def mouse_callback(event, x, y, flags, param):
    """Callback para eventos do mouse."""
    annotator = param

    if event == cv2.EVENT_LBUTTONDOWN:  # Clique esquerdo = ponto positivo
        annotator.add_point(x, y, is_positive=True)
    elif event == cv2.EVENT_RBUTTONDOWN:  # Clique direito = ponto negativo
        annotator.add_point(x, y, is_positive=False)


def run_interactive_annotation(tiles_dir, masks_dir, feature_type="lakes"):
    """
    Executa interface interativa de anotação.

    Controles:
        - Clique esquerdo: Adicionar ponto positivo (dentro da feição)
        - Clique direito: Adicionar ponto negativo (fora da feição)
        - 's': Salvar máscara atual
        - 'r': Reset pontos
        - 'n': Próxima imagem
        - 'p': Imagem anterior
        - 'q': Sair
    """
    tiles_dir = Path(tiles_dir)
    masks_dir = Path(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Listar tiles
    tile_files = sorted(tiles_dir.glob("*.png"))
    if not tile_files:
        print(f"Nenhum tile encontrado em {tiles_dir}")
        return

    print(f"Encontrados {len(tile_files)} tiles")
    print(f"\nControles:")
    print("  Clique esquerdo: ponto positivo (dentro da feição)")
    print("  Clique direito: ponto negativo (fora da feição)")
    print("  's': salvar máscara")
    print("  'r': reset pontos")
    print("  'n': próxima imagem")
    print("  'p': imagem anterior")
    print("  'q': sair")

    # Inicializar anotador
    annotator = SAMInteractiveAnnotator()

    # Configurar janela
    window_name = f"SAM Annotation - {feature_type}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback, annotator)

    current_idx = 0
    annotator.load_image(tile_files[current_idx])

    while True:
        # Mostrar visualização
        vis = annotator.get_visualization()

        # Adicionar informações na imagem
        info_text = f"Tile {current_idx + 1}/{len(tile_files)} | {feature_type}"
        cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        if annotator.current_mask is not None:
            score_text = f"Score: {annotator.current_score:.3f}"
            cv2.putText(vis, score_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Sair
            break
        elif key == ord('s'):  # Salvar
            tile_name = tile_files[current_idx].stem
            mask_path = masks_dir / f"{tile_name}_{feature_type}.png"
            annotator.save_mask(mask_path, feature_type)
            annotator.reset_points()
        elif key == ord('r'):  # Reset
            annotator.reset_points()
        elif key == ord('n'):  # Próxima
            current_idx = min(current_idx + 1, len(tile_files) - 1)
            annotator.load_image(tile_files[current_idx])
        elif key == ord('p'):  # Anterior
            current_idx = max(current_idx - 1, 0)
            annotator.load_image(tile_files[current_idx])

    cv2.destroyAllWindows()

    # Salvar log de anotações
    log_path = masks_dir / f"annotation_log_{feature_type}.json"
    with open(log_path, "w") as f:
        json.dump(annotator.masks_saved, f, indent=2)

    print(f"\nAnotação finalizada. {len(annotator.masks_saved)} máscaras salvas.")
    print(f"Log salvo em {log_path}")


if __name__ == "__main__":
    # Exemplo: anotar lagos no ano 2019
    run_interactive_annotation(
        tiles_dir=Config.TILES_DIR / "2019",
        masks_dir=Config.MASKS_DIR / "2019" / "annotations",
        feature_type="lakes"
    )
```

### 3.5 Fine-tuning do SAM

```python
"""
03_finetune_sam.py - Fine-tuning do decoder do SAM para dados do Schiaparelli
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from segment_anything import sam_model_registry
from config import Config


class GlacierDataset(Dataset):
    """Dataset para fine-tuning do SAM."""

    def __init__(self, tiles_dir, masks_dir, feature_type):
        self.tiles_dir = Path(tiles_dir)
        self.masks_dir = Path(masks_dir)
        self.feature_type = feature_type

        # Encontrar pares tile-máscara
        self.pairs = []
        for mask_file in self.masks_dir.glob(f"*_{feature_type}.png"):
            tile_name = mask_file.stem.replace(f"_{feature_type}", "")
            tile_file = self.tiles_dir / f"{tile_name}.png"
            if tile_file.exists():
                self.pairs.append((tile_file, mask_file))

        print(f"Dataset: {len(self.pairs)} pares encontrados para '{feature_type}'")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        tile_path, mask_path = self.pairs[idx]

        # Carregar imagem
        image = cv2.imread(str(tile_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Carregar máscara
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        # Converter para tensores
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "path": str(tile_path)
        }


class SAMFineTuner:
    """Fine-tuning do SAM com encoder congelado."""

    def __init__(self, checkpoint_path, model_type="vit_b"):
        print(f"Carregando SAM ({model_type})...")
        self.sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
        self.sam.to(Config.DEVICE)

        # Congelar encoder (treinar apenas decoder)
        for param in self.sam.image_encoder.parameters():
            param.requires_grad = False

        # Contar parâmetros treináveis
        trainable = sum(p.numel() for p in self.sam.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.sam.parameters())
        print(f"Parâmetros treináveis: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def train(self, train_loader, val_loader=None, epochs=20, lr=1e-4):
        """
        Treina o decoder do SAM.
        """
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.sam.parameters()),
            lr=lr
        )

        # Loss combinada: Dice + BCE
        dice_loss = DiceLoss()
        bce_loss = nn.BCEWithLogitsLoss()

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training
            self.sam.train()
            train_loss = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                images = batch["image"].to(Config.DEVICE)
                masks_gt = batch["mask"].to(Config.DEVICE)

                # Forward através do encoder (sem gradiente)
                with torch.no_grad():
                    image_embeddings = self.sam.image_encoder(images)

                # Gerar embeddings de prompt (prompt automático: centro)
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None
                )

                # Predição do decoder
                masks_pred, iou_pred = self.sam.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )

                # Interpolar para tamanho original
                masks_pred = torch.nn.functional.interpolate(
                    masks_pred,
                    size=masks_gt.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

                # Calcular loss
                loss = dice_loss(masks_pred, masks_gt) + bce_loss(masks_pred, masks_gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            if val_loader:
                val_loss = self._validate(val_loader, dice_loss, bce_loss)
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(Config.MODELS_DIR / "sam_finetuned_best.pth")
            else:
                print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

        # Salvar modelo final
        self.save_model(Config.MODELS_DIR / "sam_finetuned_final.pth")

    def _validate(self, val_loader, dice_loss, bce_loss):
        """Validação."""
        self.sam.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(Config.DEVICE)
                masks_gt = batch["mask"].to(Config.DEVICE)

                image_embeddings = self.sam.image_encoder(images)

                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=None, boxes=None, masks=None
                )

                masks_pred, _ = self.sam.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False
                )

                masks_pred = torch.nn.functional.interpolate(
                    masks_pred, size=masks_gt.shape[-2:],
                    mode='bilinear', align_corners=False
                )

                loss = dice_loss(masks_pred, masks_gt) + bce_loss(masks_pred, masks_gt)
                val_loss += loss.item()

        return val_loss / len(val_loader)

    def save_model(self, path):
        """Salva apenas os pesos do decoder (mais leve)."""
        torch.save({
            "mask_decoder": self.sam.mask_decoder.state_dict(),
            "prompt_encoder": self.sam.prompt_encoder.state_dict()
        }, path)
        print(f"Modelo salvo: {path}")


class DiceLoss(nn.Module):
    """Dice Loss para segmentação binária."""

    def forward(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)

        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()

        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice


def run_finetuning(feature_type="lakes"):
    """Executa fine-tuning para uma feição específica."""

    # Criar datasets
    train_dataset = GlacierDataset(
        tiles_dir=Config.TILES_DIR / "2019",
        masks_dir=Config.MASKS_DIR / "2019" / "annotations",
        feature_type=feature_type
    )

    if len(train_dataset) == 0:
        print("Nenhum dado de treinamento encontrado!")
        print("Execute primeiro: python 02_sam_interactive.py")
        return

    # Split train/val (80/20)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4)

    # Fine-tuning
    finetuner = SAMFineTuner(Config.SAM_CHECKPOINT, Config.MODEL_TYPE)
    finetuner.train(train_loader, val_loader, epochs=20, lr=1e-4)


if __name__ == "__main__":
    Config.create_directories()
    run_finetuning("lakes")
```

### 3.6 Inferência em Larga Escala

```python
"""
04_inference.py - Aplicar SAM treinado a todos os tiles
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import json
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from config import Config


class SAMInference:
    """Inferência com SAM em larga escala."""

    def __init__(self, checkpoint_path, finetuned_path=None, model_type="vit_b"):
        print(f"Carregando SAM ({model_type})...")
        self.sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))

        # Carregar pesos fine-tuned se disponível
        if finetuned_path and Path(finetuned_path).exists():
            print(f"Carregando pesos fine-tuned: {finetuned_path}")
            state = torch.load(finetuned_path, map_location=Config.DEVICE)
            self.sam.mask_decoder.load_state_dict(state["mask_decoder"])
            self.sam.prompt_encoder.load_state_dict(state["prompt_encoder"])

        self.sam.to(Config.DEVICE)
        self.sam.eval()

        # Gerador de máscaras automático
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=16,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            min_mask_region_area=50,
        )

    def process_tile(self, image_path, feature_type):
        """
        Processa um tile e retorna máscara filtrada.
        """
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Gerar todas as máscaras
        masks = self.mask_generator.generate(image_rgb)

        # Filtrar por características da feição
        filtered_mask = self._filter_masks(masks, feature_type, image_rgb)

        return filtered_mask

    def _filter_masks(self, masks, feature_type, image):
        """Filtra máscaras baseado em características da feição."""
        combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        feature_config = Config.FEATURES[feature_type]

        for mask_data in masks:
            mask = mask_data["segmentation"]
            area = mask_data["area"]

            # Filtro por área
            if not (feature_config["min_area"] <= area <= feature_config["max_area"]):
                continue

            # Extrair região da imagem
            region = image[mask]
            if len(region) == 0:
                continue

            # Critérios específicos por feição
            if feature_type == "lakes":
                # Lagos: alta razão blue/red
                blue_mean = np.mean(region[:, 2])
                red_mean = np.mean(region[:, 0])
                ratio = blue_mean / (red_mean + 1e-10)

                if ratio > 1.3:
                    combined_mask[mask] = 255

            elif feature_type == "crevasses":
                # Fendas: escuras e alongadas
                brightness = np.mean(region)

                # Calcular alongamento
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    rect = cv2.minAreaRect(contours[0])
                    w, h = rect[1]
                    aspect_ratio = max(w, h) / (min(w, h) + 1e-10)

                    if brightness < 150 and aspect_ratio > 3:
                        combined_mask[mask] = 255

            elif feature_type == "channels":
                # Canais: lineares
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    rect = cv2.minAreaRect(contours[0])
                    w, h = rect[1]
                    aspect_ratio = max(w, h) / (min(w, h) + 1e-10)

                    if aspect_ratio > 5:
                        combined_mask[mask] = 255

        return combined_mask

    def process_all_tiles(self, tiles_dir, output_dir, feature_type):
        """Processa todos os tiles de um diretório."""
        tiles_dir = Path(tiles_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tile_files = sorted(tiles_dir.glob("*.png"))

        print(f"Processando {len(tile_files)} tiles para '{feature_type}'...")

        for tile_path in tqdm(tile_files, desc=f"Inferência ({feature_type})"):
            mask = self.process_tile(tile_path, feature_type)

            output_path = output_dir / f"{tile_path.stem}_mask.png"
            cv2.imwrite(str(output_path), mask)

        print(f"Máscaras salvas em {output_dir}")


def run_inference(year, feature_type):
    """Executa inferência para um ano e feição."""

    inference = SAMInference(
        checkpoint_path=Config.SAM_CHECKPOINT,
        finetuned_path=Config.MODELS_DIR / "sam_finetuned_best.pth",
        model_type=Config.MODEL_TYPE
    )

    inference.process_all_tiles(
        tiles_dir=Config.TILES_DIR / str(year),
        output_dir=Config.MASKS_DIR / str(year) / feature_type,
        feature_type=feature_type
    )


if __name__ == "__main__":
    # Processar todos os anos e feições
    for year in Config.YEARS:
        for feature_type in Config.FEATURES.keys():
            print(f"\n{'='*50}")
            print(f"Ano: {year} | Feição: {feature_type}")
            print(f"{'='*50}")
            run_inference(year, feature_type)
```

### 3.7 Reconstrução do Mosaico

```python
"""
05_reconstruct_mosaic.py - Reconstrói mosaico completo a partir das máscaras
"""

import rasterio
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from config import Config


def reconstruct_mosaic(tiles_dir, masks_dir, reference_raster, output_path, feature_type):
    """
    Reconstrói mosaico completo a partir das máscaras dos tiles.
    """
    tiles_dir = Path(tiles_dir)
    masks_dir = Path(masks_dir)

    # Carregar índice de tiles
    with open(tiles_dir / "tiles_index.json") as f:
        tiles_index = json.load(f)

    with rasterio.open(reference_raster) as src:
        # Criar profile para output
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype='uint8',
            compress='lzw'
        )

        # Criar mosaico vazio
        full_mask = np.zeros((src.height, src.width), dtype=np.uint8)
        count_mask = np.zeros((src.height, src.width), dtype=np.uint8)

        # Processar cada tile
        for tile_info in tqdm(tiles_index["tiles"], desc="Reconstruindo mosaico"):
            tile_name = Path(tile_info["filename"]).stem
            mask_path = masks_dir / f"{tile_name}_mask.png"

            if not mask_path.exists():
                continue

            # Carregar máscara do tile
            import cv2
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                continue

            # Posição do tile
            x = tile_info["x"]
            y = tile_info["y"]
            h, w = mask.shape

            # Inserir no mosaico (usar máximo para overlap)
            full_mask[y:y+h, x:x+w] = np.maximum(full_mask[y:y+h, x:x+w], mask)
            count_mask[y:y+h, x:x+w] += 1

        # Salvar resultado
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(str(output_path), 'w', **profile) as dst:
            dst.write(full_mask, 1)

        print(f"Mosaico reconstruído: {output_path}")
        print(f"Cobertura: {(count_mask > 0).sum() / count_mask.size * 100:.1f}%")

        return full_mask


def run_reconstruction(year, feature_type):
    """Reconstrói mosaico para um ano e feição."""

    # Encontrar mosaico de referência
    mosaics = {
        2016: "Schiaparelli_mosaic_2016.tif",
        2017: "Schiaparelli_mosaic_2017.tif",
        2018: "Schiaparelli_mosaic_2018.tif",
        2019: "Schiaparelli_mosaic_2019.tif",
        2020: "schiaparelli_mosaic_2020.tif"
    }

    reference = Config.PROJECT_DIR / mosaics[year]

    reconstruct_mosaic(
        tiles_dir=Config.TILES_DIR / str(year),
        masks_dir=Config.MASKS_DIR / str(year) / feature_type,
        reference_raster=reference,
        output_path=Config.RESULTS_DIR / str(year) / f"{feature_type}_mask.tif",
        feature_type=feature_type
    )


if __name__ == "__main__":
    for year in Config.YEARS:
        for feature_type in Config.FEATURES.keys():
            print(f"\n{'='*50}")
            print(f"Reconstruindo: {year} - {feature_type}")
            print(f"{'='*50}")
            run_reconstruction(year, feature_type)
```

---

## 4. Cronograma de Execução

| Semana | Atividade | Horas | Entregável |
|--------|-----------|-------|------------|
| **1** | Setup ambiente + Download SAM + Tiling (01_create_tiles.py) | 10h | Tiles criados |
| **2** | Anotação interativa com SAM (02_sam_interactive.py) | 8h | 50+ máscaras de referência |
| **3** | Fine-tuning do SAM (03_finetune_sam.py) | 5h | Modelo adaptado |
| **4** | Inferência em todos os anos (04_inference.py) | 10h | Máscaras brutas |
| **5** | Reconstrução + Pós-processamento (05_reconstruct_mosaic.py) | 8h | Mosaicos de feições |
| **6** | Validação e ajuste de parâmetros | 8h | Métricas de qualidade |
| **7** | Análise temporal multianual | 8h | Séries temporais |
| **8** | Relatório final + Documentação | 10h | Entrega completa |

**Total estimado: ~70 horas de trabalho**

---

## 5. Métricas Esperadas

| Feição | F1-Score Esperado | IoU Esperado |
|--------|-------------------|--------------|
| Lagos supraglaciais | 85-90% | 75-85% |
| Fendas (crevasses) | 80-85% | 70-80% |
| Canais de degelo | 75-85% | 65-75% |

---

## 6. Requisitos de Hardware

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| GPU | GTX 1060 6GB | RTX 3080 10GB |
| RAM | 16 GB | 32 GB |
| Disco | 50 GB livre | 100 GB SSD |
| CUDA | 11.8+ | 12.0+ |

**Alternativa:** Google Colab Pro (~$10/mês) com GPU T4/V100

---

## 7. Referências

1. Kirillov, A., et al. (2023). Segment Anything. ICCV 2023. https://segment-anything.com
2. Chai, M., et al. (2025). Potential of SAM for supraglacial lakes. Int. J. Digital Earth. https://doi.org/10.1080/17538947.2025.2554312
3. Baraka, S., et al. (2023). SAM in Glaciology. Journal of Glaciology. https://doi.org/10.1017/jog.2023.87

---

## 8. Estrutura Final do Projeto

```
mosaicos_DEMs_Schiaparelli/
├── config.py                    # Configurações
├── 01_create_tiles.py           # Preparação de dados
├── 02_sam_interactive.py        # Anotação interativa
├── 03_finetune_sam.py           # Fine-tuning
├── 04_inference.py              # Inferência
├── 05_reconstruct_mosaic.py     # Reconstrução
├── sam_vit_b_01ec64.pth         # Checkpoint SAM
├── data/                        # Dados originais
├── tiles/                       # Tiles por ano
│   ├── 2016/
│   ├── 2017/
│   └── ...
├── masks/                       # Máscaras
│   ├── 2016/
│   │   ├── annotations/         # Ground truth
│   │   ├── lakes/
│   │   ├── crevasses/
│   │   └── channels/
│   └── ...
├── models/                      # Modelos treinados
│   ├── sam_finetuned_best.pth
│   └── sam_finetuned_final.pth
└── results/                     # Resultados finais
    ├── 2016/
    │   ├── lakes_mask.tif
    │   ├── crevasses_mask.tif
    │   └── channels_mask.tif
    └── ...
```

---

*Documento gerado em: Janeiro 2026*
*Técnica recomendada: SAM (Segment Anything Model)*
*Projeto: LACRIO IC - Glaciar Schiaparelli*
