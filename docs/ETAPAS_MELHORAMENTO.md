# Etapas de Melhoramento do Pipeline SAM

**Projeto:** LACRIO IC - Extracao de Feicoes Supraglaciais (Glaciar Schiaparelli)
**Data:** 2026-02-08
**Baseline:** F1 lakes=0.26 | crevasses=0.29 | channels=0.45
**Metas:** F1 lakes=0.85-0.90 | crevasses=0.80-0.85 | channels=0.75-0.85

---

## Visao geral

```
Baseline (atual)          Etapa 1-2          Etapa 3-4          Etapa 5-8
F1 ~0.26-0.45    -->    F1 ~0.35-0.55   -->  F1 ~0.50-0.70  -->  F1 ~0.60-0.80+
                  Loss + SAM-HQ      Augmentation + LoRA     TTA + CRF + filtros
```

Cada etapa e independente e pode ser validada antes de avancar para a proxima.

---

## Etapa 1 -- Focal Tversky Loss por feicao

**Impacto esperado:** +3-5% IoU
**Esforco:** Baixo (~1-2 horas)
**Arquivo:** `03_finetune_sam.py`

### Problema
A loss atual (Dice + BCE) trata falsos positivos e falsos negativos com peso igual. Lakes sofrem com FP (sombras detectadas como agua) enquanto crevasses/channels sofrem com FN (feicoes perdidas).

### O que fazer

1. Implementar `FocalTverskyLoss` com parametros alpha/beta configuraveis por feicao:

```python
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0):
        super().__init__()
        self.alpha = alpha   # peso para falsos positivos
        self.beta = beta     # peso para falsos negativos
        self.gamma = gamma   # fator de focalizacao (< 1 foca em hard examples)
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky) ** self.gamma
```

2. Usar pesos diferentes por feicao:

| Feicao | alpha (peso FP) | beta (peso FN) | Justificativa |
|---|---|---|---|
| lakes | 0.6 | 0.4 | Penalizar mais FP (sombras) |
| crevasses | 0.3 | 0.7 | Penalizar mais FN (recall) |
| channels | 0.3 | 0.7 | Penalizar mais FN (recall) |

3. Combinar com Focal Loss:

```python
class ImprovedLoss(nn.Module):
    def __init__(self, feature="lakes"):
        super().__init__()
        if feature == "lakes":
            self.tversky = FocalTverskyLoss(alpha=0.6, beta=0.4)
        else:
            self.tversky = FocalTverskyLoss(alpha=0.3, beta=0.7)
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)

    def forward(self, pred, target):
        return 0.5 * self.tversky(pred, target) + 0.5 * self.focal(pred, target)
```

4. Passar o nome da feicao para `train_feature()` e usar a loss correspondente.

### Validacao
- Re-treinar cada feicao com a nova loss (30 epocas).
- Comparar metricas com a baseline usando `06_validate.py`.
- Esperar melhora em precision para lakes e recall para crevasses/channels.

### Referencia
- Salehi et al. (2017) "Tversky loss function for image segmentation"
- Abraham & Khan (2019) "A novel focal Tversky loss function with improved attention U-Net"

---

## Etapa 2 -- SAM-HQ (drop-in replacement)

**Impacto esperado:** +3-5% IoU
**Esforco:** Baixo (~1-2 horas)
**Arquivos:** `03_finetune_sam.py`, `04_inference.py`

### Problema
O SAM vanilla produz mascaras com bordas imprecisas, especialmente para features finas como crevasses e channels. Isso causa FP nas bordas e reduz IoU.

### O que fazer

1. Instalar SAM-HQ:

```bash
pip install segment-anything-hq
# ou
git clone https://github.com/SysCV/sam-hq.git
```

2. Baixar checkpoint SAM-HQ ViT-B:

```bash
wget https://huggingface.co/lkeab/hq-sam/resolve/main/sam_hq_vit_b.pth
```

3. Trocar imports nos scripts:

```python
# Antes
from segment_anything import sam_model_registry
# Depois
from segment_anything_hq import sam_model_registry
```

4. Atualizar `Config.SAM_CHECKPOINT` para o novo checkpoint.

5. No decoder, habilitar o HQ output token:

```python
# Na inferencia, usar o output HQ
masks_pred, iou_pred = sam.mask_decoder(
    image_embeddings=image_embedding,
    image_pe=image_pe,
    sparse_prompt_embeddings=sparse_emb,
    dense_prompt_embeddings=dense_emb,
    multimask_output=False,
    hq_token_only=True,  # <-- adicionar isto
)
```

### Validacao
- Rodar inferencia com `--annotated-only` para comparar rapidamente.
- Esperar melhora principal em crevasses e channels (bordas finas).
- Se o resultado for bom, re-treinar o decoder HQ do zero (30 epocas).

### Referencia
- Ke et al. (2023) "Segment Anything in High Quality" (NeurIPS 2023)

---

## Etapa 3 -- Data augmentation com Albumentations

**Impacto esperado:** +5-8% IoU
**Esforco:** Medio (~4-6 horas)
**Arquivo:** `03_finetune_sam.py`

### Problema
O treino atual nao aplica nenhuma augmentacao. Com apenas ~50 amostras positivas por feicao, o modelo overfita rapidamente (melhor epoca = 5-7 de 30).

### Pre-requisito
Esta etapa requer mudar de embeddings pre-computados para encoding on-the-fly. Isso e necessario porque as augmentacoes devem ser aplicadas na imagem antes do encoder.

### O que fazer

1. Instalar Albumentations:

```bash
pip install albumentations
```

2. Definir pipeline de augmentacao:

```python
import albumentations as A

train_transform = A.Compose([
    # Geometricas (aplicar na imagem + mascara)
    A.RandomRotate90(p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),

    # Espectrais (aplicar apenas na imagem)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.RandomGamma(gamma_limit=(70, 150), p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.3),
])
```

3. Modificar `GlacierEmbeddingDataset` para `GlacierOnTheFlyDataset`:

```python
class GlacierOnTheFlyDataset(Dataset):
    def __init__(self, tile_paths, mask_paths, transform=None):
        self.tile_paths = tile_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __getitem__(self, idx):
        image = cv2.imread(str(self.tile_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mask_paths[idx] is not None:
            mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Resize para 1024x1024 (entrada SAM)
        image = cv2.resize(image, (1024, 1024))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        image_tensor = normalize_image(image_tensor)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor
```

4. No loop de treino, rodar o encoder on-the-fly (float16) dentro do `torch.no_grad()` e treinar apenas o decoder.

### Augmentacao avancada: Copy-Paste

Para feicoes raras, implementar copy-paste augmentation:
- Recortar regioes anotadas (lago, fenda) de um tile.
- Colar em outro tile de fundo (sem feicao).
- Gerar mascara combinada.
- Efetivamente dobra/triplica o dataset de positivos.

### Validacao
- Comparar curvas de treino (val loss por epoca) com e sem augmentacao.
- Esperar: melhor epoca mais tardia (menos overfit), val loss final menor.
- Comparar F1/IoU com `06_validate.py`.

### Referencia
- Ghiasi et al. (2021) "Simple Copy-Paste is a Strong Data Augmentation Method"

---

## Etapa 4 -- LoRA no encoder (fine-tuning parcial)

**Impacto esperado:** +5-10% IoU
**Esforco:** Medio (~4-6 horas)
**Arquivo:** `03_finetune_sam.py`

### Problema
Com o encoder congelado, o modelo nao consegue aprender features visuais especificas do dominio glacial (textura de gelo, assinatura espectral de agua em RGB, padroes de sombra). O domain gap entre imagens naturais (SA-1B) e imagens glaciais de drone e grande.

### Pre-requisito
Requer a mudanca da Etapa 3 (encoding on-the-fly em vez de embeddings pre-computados).

### O que fazer

1. Instalar PEFT:

```bash
pip install peft
```

2. Injetar LoRA no encoder:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=4,                              # rank baixo (4-8)
    lora_alpha=16,                    # fator de escala
    target_modules=["qkv"],           # aplicar nas projecoes de atencao
    lora_dropout=0.1,
    bias="none",
)

# Aplicar LoRA ao encoder
sam.image_encoder = get_peft_model(sam.image_encoder, lora_config)

# Agora treinar: decoder params + LoRA params + prompt encoder params
trainable_params = (
    list(sam.mask_decoder.parameters()) +
    list(sam.prompt_encoder.parameters()) +
    [p for p in sam.image_encoder.parameters() if p.requires_grad]
)
```

3. Usar learning rate diferenciada:

```python
optimizer = torch.optim.AdamW([
    {"params": list(sam.mask_decoder.parameters()), "lr": 1e-4},
    {"params": list(sam.prompt_encoder.parameters()), "lr": 1e-4},
    {"params": [p for p in sam.image_encoder.parameters() if p.requires_grad], "lr": 1e-5},
], weight_decay=1e-4)
```

4. Parametros adicionais LoRA: apenas ~0.3M (vs ~4M do decoder). Total treinavel: ~4.3M. Cabe em 4GB VRAM com batch_size=1 e gradient checkpointing se necessario.

### Alternativa sem PEFT (manual)
Se `peft` der problemas com a arquitetura SAM:

```python
class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=4, alpha=16):
        super().__init__()
        self.original = original_layer
        d = original_layer.in_features
        self.lora_A = nn.Linear(d, r, bias=False)
        self.lora_B = nn.Linear(r, original_layer.out_features, bias=False)
        self.scale = alpha / r
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        self.original.weight.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(x)) * self.scale
```

### Validacao
- Comparar Val Dice ao longo das epocas vs decoder-only.
- Esperar: convergencia mais lenta mas Val Dice final significativamente maior.
- Monitorar VRAM com `nvidia-smi` para garantir que cabe em 4GB.

### Referencia
- Hu et al. (2022) "LoRA: Low-Rank Adaptation of Large Language Models"
- Chen et al. (2024) "SAM-Adapter: Adapting SAM for Medical Image Segmentation"

---

## Etapa 5 -- Test-Time Augmentation (TTA)

**Impacto esperado:** +2-4% IoU
**Esforco:** Baixo (~2-3 horas)
**Arquivo:** `04_inference.py`

### O que fazer

1. Modificar `predict_tile` para retornar probabilidades (antes de binarizar):

```python
def predict_tile_proba(sam, image, pred_iou_threshold=0.5, combine_mode="max"):
    """Retorna mapa de probabilidade float32 (256x256) em vez de mascara binaria."""
    # ... (mesmo codigo atual ate combined_mask)
    return combined_mask  # float32, sem binarizar
```

2. Implementar TTA 4-fold:

```python
def predict_with_tta(sam, image, threshold=0.5, pred_iou_threshold=0.5, combine_mode="max"):
    h, w = image.shape[:2]
    probs = []

    transforms = [
        (lambda x: x,                    lambda x: x),                      # original
        (lambda x: np.fliplr(x).copy(),  lambda x: np.fliplr(x).copy()),   # hflip
        (lambda x: np.flipud(x).copy(),  lambda x: np.flipud(x).copy()),   # vflip
        (lambda x: np.rot90(x, 2).copy(), lambda x: np.rot90(x, -2).copy()), # 180
    ]

    for fwd, inv in transforms:
        aug_image = fwd(image)
        prob = predict_tile_proba(sam, aug_image, pred_iou_threshold, combine_mode)
        prob_back = inv(prob)
        prob_resized = cv2.resize(prob_back, (w, h), interpolation=cv2.INTER_LINEAR)
        probs.append(prob_resized)

    avg_prob = np.mean(probs, axis=0)
    return (avg_prob > threshold).astype(np.uint8) * 255
```

3. Adicionar flag `--tta` ao argparse de `04_inference.py`.

### Validacao
- Comparar F1/IoU com e sem TTA no modo `--annotated-only`.
- Custo: 4x mais lento (~12 min vs ~3 min para tiles anotados).

---

## Etapa 6 -- Refinamento iterativo com mask prompt (2-pass)

**Impacto esperado:** +2-3% IoU
**Esforco:** Baixo (~1-2 horas)
**Arquivo:** `04_inference.py`

### Problema
Atualmente cada ponto de prompt gera uma mascara independente (`masks=None` no prompt encoder). O SAM tem capacidade built-in de refinar mascaras usando uma mascara anterior como input.

### O que fazer

Modificar `predict_tile` para fazer 2 passes:

```python
# Pass 1: gerar mascara inicial (como atualmente)
initial_mask = predict_tile_proba(sam, image, ...)

# Pass 2: usar mascara inicial como prompt adicional
mask_input = torch.from_numpy(initial_mask).unsqueeze(0).unsqueeze(0).float()
mask_input = mask_input.to(Config.DEVICE)
mask_input = F.interpolate(mask_input, size=(256, 256), mode="bilinear")

sparse_emb, dense_emb = sam.prompt_encoder(
    points=(best_point, best_label),
    boxes=None,
    masks=mask_input,  # <-- mascara anterior como prompt
)

refined_mask, _ = sam.mask_decoder(
    image_embeddings=image_embedding,
    image_pe=image_pe,
    sparse_prompt_embeddings=sparse_emb,
    dense_prompt_embeddings=dense_emb,
    multimask_output=False,
)
```

### Validacao
- Comparar IoU por tile antes e depois do refinamento.
- Esperar melhora mais forte em features com bordas complexas (channels).

---

## Etapa 7 -- Filtro de slope do DEM para lakes

**Impacto esperado:** Reducao de 5-10% dos FP em lakes
**Esforco:** Baixo (~2 horas)
**Arquivos:** `shadow_utils.py`, `04_inference.py`

### Problema
Lagos supraglaciais nao existem em encostas ingremes (a agua escorre). Deteccoes em slopes > 15 graus sao quase certamente FP.

### O que fazer

1. Adicionar funcao em `shadow_utils.py`:

```python
def compute_slope_map(dem, res_x, res_y):
    """Computa slope em graus a partir do DEM."""
    dy, dx = np.gradient(dem, res_y, res_x)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg
```

2. Pre-computar slope junto com shadow em `precompute_year_shadows()`.

3. Em `04_inference.py`, rejeitar deteccoes de lakes onde slope medio > 15 graus:

```python
if feature == "lakes" and slope_map is not None:
    tile_slope = get_slope_for_tile(tile_info, slope_map, dem_transform)
    # Remover componentes em areas ingremes
    mask = filter_by_slope(mask, tile_slope, max_slope=15.0)
```

### Validacao
- Contar quantos FP sao removidos no modo `--annotated-only`.
- Verificar que nenhum TP e removido (lagos reais em areas planas).

---

## Etapa 8 -- CRF pos-processamento

**Impacto esperado:** +1-3% IoU
**Esforco:** Baixo (~2 horas)
**Arquivo:** `04_inference.py`

### O que fazer

1. Instalar:

```bash
pip install pydensecrf
```

2. Aplicar CRF apos a mascara bruta, antes dos filtros de feicao:

```python
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def apply_crf(image, prob_map, n_iters=5, sxy=80, srgb=20):
    h, w = image.shape[:2]
    probs = np.stack([1 - prob_map, prob_map], axis=0)
    unary = unary_from_softmax(probs)

    d = dcrf.DenseCRF2D(w, h, 2)
    d.setUnaryEnergy(unary)
    d.addPairwiseBilateral(sxy=sxy, srgb=srgb, rgbim=image.copy(order='C'), compat=10)
    d.addPairwiseGaussian(sxy=3, compat=3)

    Q = d.inference(n_iters)
    result = np.argmax(Q, axis=0).reshape(h, w)
    return (result * 255).astype(np.uint8)
```

3. Parametros recomendados para glaciologia:
   - `sxy=80-100` (features glaciais sao mais suaves)
   - `srgb=20-30` (diferencas espectrais sutis entre agua e gelo)

### Validacao
- Comparar bordas das mascaras antes/depois do CRF visualmente.
- Medir IoU com `06_validate.py`.

---

## Etapas futuras (apos atingir F1 > 0.60)

### Etapa 9 -- Cross-validation 5-fold
- Substituir split unico 80/20 por 5-fold CV.
- Reportar media +/- desvio padrao das metricas.
- Garante estimativa confiavel com dataset pequeno.

### Etapa 10 -- Pseudo-labeling / Self-training
- Treinar modelo com dados anotados.
- Rodar inferencia em todos os ~22k tiles.
- Selecionar predicoes com IoU predito > 0.9.
- Adicionar como pseudo-labels e re-treinar.
- Multiplica dataset efetivo por 2-5x.

### Etapa 11 -- Multi-temporal consistency
- Cruzar deteccoes entre 2016-2020.
- Features que aparecem em apenas 1 ano = provavel FP.
- Lagos reais tendem a reaparecer nas mesmas depressoes.

### Etapa 12 -- Boundary Loss para features lineares
- Adicionar termo de loss baseado em distance transform.
- Penaliza predicoes longe das bordas GT.
- Especialmente util para crevasses e channels.

### Etapa 13 -- Pre-treino em datasets publicos
- CALFIN (~1800 imagens de frentes de calvagem).
- GlacierNet2 (segmentacao de geleiras).
- Transferir features glaciologicas antes do fine-tuning no Schiaparelli.

### Etapa 14 -- Migrar para SAM2 Hiera-Tiny
- Backbone hierarquico mais eficiente.
- Multi-scale features built-in.
- Requer refatoracao maior que SAM-HQ.

---

## Checklist de progresso

- [x] Etapa 1: Focal Tversky Loss
- [x] Etapa 2: SAM-HQ
- [x] Etapa 3: Data augmentation
- [x] Etapa 4: LoRA no encoder
- [ ] Etapa 5: TTA na inferencia
- [ ] Etapa 6: Mask refinement 2-pass
- [ ] Etapa 7: Slope filter (DEM)
- [ ] Etapa 8: CRF pos-processamento
- [ ] Etapa 9: Cross-validation 5-fold
- [ ] Etapa 10: Pseudo-labeling
- [ ] Etapa 11: Multi-temporal consistency
- [ ] Etapa 12: Boundary Loss
- [ ] Etapa 13: Pre-treino em datasets publicos
- [ ] Etapa 14: SAM2 Hiera-Tiny

---

## Referencias

1. Salehi et al. (2017) "Tversky loss function for image segmentation using 3D FCDN"
2. Abraham & Khan (2019) "A novel focal Tversky loss function with improved attention U-Net"
3. Ke et al. (2023) "Segment Anything in High Quality" NeurIPS 2023
4. Hu et al. (2022) "LoRA: Low-Rank Adaptation of Large Language Models"
5. Chen et al. (2024) "SAM-Adapter: Adapting SAM for Medical Image Segmentation"
6. Ghiasi et al. (2021) "Simple Copy-Paste is a Strong Data Augmentation Method"
7. Krahenbuhl & Koltun (2011) "Efficient Inference in Fully Connected CRFs"
8. Kervadec et al. (2019) "Boundary loss for highly unbalanced segmentation" MIDL
9. Ravi et al. (2024) "SAM 2: Segment Anything in Images and Videos"
10. Lin et al. (2017) "Focal Loss for Dense Object Detection"
11. Chai et al. (2025) "Potential of SAM for supraglacial lakes" Int. J. Digital Earth
