"""
config.py - Configurações do projeto SAM para Glaciar Schiaparelli

Projeto: LACRIO IC - Extração de Feições Supraglaciais
Data: Janeiro 2026
"""

from pathlib import Path

# Torch é opcional para scripts que não precisam de GPU (ex: tiling)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _detect_project_dir() -> Path:
    """
    Detecta automaticamente o diretório raiz do projeto.
    """
    return Path(__file__).resolve().parent


def _detect_data_source_dir(project_dir: Path) -> Path:
    """
    Detecta automaticamente a pasta dos dados brutos.
    """
    candidates = [
        project_dir / "Schiaparelli_glacier",
        project_dir / "mosaicos_DEMs_Schiaparelli",
    ]
    for path in candidates:
        if path.exists():
            return path
    # fallback para o caminho esperado atual
    return project_dir / "Schiaparelli_glacier"

class Config:
    """Configurações centralizadas do projeto."""
    
    # =========================================================================
    # DIRETÓRIOS
    # =========================================================================
    PROJECT_DIR = _detect_project_dir()
    DATA_SOURCE_DIR = _detect_data_source_dir(PROJECT_DIR)  # Onde estão os mosaicos .tif
    DATA_DIR = PROJECT_DIR / "data"
    TILES_DIR = PROJECT_DIR / "tiles"
    MASKS_DIR = PROJECT_DIR / "masks"
    MODELS_DIR = PROJECT_DIR / "models"
    RESULTS_DIR = PROJECT_DIR / "results"
    
    # =========================================================================
    # MODELO SAM
    # =========================================================================
    # Opções: "vit_b" (rápido), "vit_l" (médio), "vit_h" (preciso)
    MODEL_TYPE = "vit_b"

    # SAM-HQ: mascaras de maior qualidade, especialmente bordas finas
    # Ref: Ke et al. (2023) "Segment Anything in High Quality" NeurIPS
    USE_SAM_HQ = True   # SAM-HQ ativado

    # Checkpoints
    SAM_CHECKPOINT_STANDARD = PROJECT_DIR / "sam_vit_b_01ec64.pth"
    SAM_HQ_CHECKPOINT = PROJECT_DIR / "sam_hq_vit_b.pth"

    # Selecao automatica do checkpoint
    SAM_CHECKPOINT = SAM_HQ_CHECKPOINT if USE_SAM_HQ else SAM_CHECKPOINT_STANDARD
    
    # Device (CUDA se disponível)
    DEVICE = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    
    # =========================================================================
    # TILING
    # =========================================================================
    TILE_SIZE = 512      # Tamanho do tile em pixels
    OVERLAP = 64         # Sobreposição entre tiles
    MIN_VALID_RATIO = 0.7  # Mínimo de pixels válidos (não NoData)
    
    # =========================================================================
    # MOSAICOS DISPONÍVEIS
    # =========================================================================
    MOSAICS = {
        2016: "Schiaparelli_mosaic_2016.tif",
        2017: "Schiaparelli_mosaic_2017.tif",
        2018: "Schiaparelli_mosaic_2018.tif",
        2019: "Schiaparelli_mosaic_2019.tif",
        2020: "schiaparelli_mosaic_2020.tif"
    }
    
    YEARS = list(MOSAICS.keys())
    
    # =========================================================================
    # DEMs DISPONÍVEIS
    # =========================================================================
    DEMS = {
        2016: "Schiaparelli_DEM_2016.tif",
        2017: "Schiaparelli_DEM_2017.tif",
        2018: "Schiaparelli_DEM_2018.tif",
        2019: "Schiaparelli_DEM_2019.tif",
        2020: "schiaparelli_DEM_2020.tif",
        2022: "schiaparelli_DEM_2022.tif"
    }
    
    # =========================================================================
    # FEIÇÕES ALVO
    # =========================================================================
    FEATURES = {
        "lakes": {
            "description": "Lagos e poças supraglaciais",
            "color": (0, 0, 255),      # Azul (BGR)
            "color_rgb": (0, 0, 255),  # Azul (RGB)
            "min_area": 20,            # Área mínima em pixels (reduzido para captar poças)
            "max_area": 100000         # Área máxima em pixels (ampliado para lagos grandes)
        },
        "crevasses": {
            "description": "Fendas no gelo",
            "color": (0, 0, 255),      # Vermelho (BGR)
            "color_rgb": (255, 0, 0),  # Vermelho (RGB)
            "min_area": 100,
            "max_area": 10000
        },
        "channels": {
            "description": "Canais de água de degelo",
            "color": (255, 255, 0),    # Ciano (BGR)
            "color_rgb": (0, 255, 255),# Ciano (RGB)
            "min_area": 200,
            "max_area": 20000
        }
    }
    
    # =========================================================================
    # NORMALIZAÇÃO SAM (ImageNet stats usadas pelo ViT pré-treinado)
    # =========================================================================
    PIXEL_MEAN = [123.675, 116.28, 103.53]
    PIXEL_STD = [58.395, 57.12, 57.375]

    # =========================================================================
    # TREINAMENTO
    # =========================================================================
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5           # Reduzido de 1e-4 para evitar colapso do decoder
    EPOCHS = 50                    # Mais epocas com LR menor
    TRAIN_VAL_SPLIT = 0.8

    # Warmup: LR cresce linearmente de 0 ate LEARNING_RATE nas primeiras epocas
    WARMUP_EPOCHS = 5

    # Early stopping: para o treino se val_dice nao melhora por N epocas
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.001  # melhoria minima para considerar progresso

    # =========================================================================
    # DATA AUGMENTATION (Etapa 3)
    # =========================================================================
    # Ativar augmentacao requer encoding on-the-fly (encoder roda durante treino)
    # VRAM: ~3 GB com batch_size=1 (encoder fp16 + decoder fp32)
    USE_AUGMENTATION = True         # Ativado por padrao (melhoria comprovada)

    # Copy-Paste Augmentation: recorta feicoes de tiles positivos e cola em
    # tiles negativos, multiplicando o dataset efetivo por 2-3x.
    # Ref: Ghiasi et al. (2021) "Simple Copy-Paste is a Strong Data Augmentation"
    USE_COPY_PASTE = True
    COPY_PASTE_PROB = 0.5           # probabilidade de aplicar copy-paste por amostra

    # =========================================================================
    # LoRA - Low-Rank Adaptation do encoder (Etapa 4)
    # =========================================================================
    # Injeta matrizes de baixo rank nas projecoes QKV do ViT encoder
    # Parametros extras: ~0.3M (vs ~4M decoder). Total ~4.3M treinavel.
    # Pre-requisito: USE_AUGMENTATION=True (encoding on-the-fly)
    # VRAM: ~3-4 GB com batch_size=1
    USE_LORA = False
    LORA_RANK = 4              # rank das matrizes LoRA (4-8 recomendado)
    LORA_ALPHA = 16            # fator de escala (alpha/r)
    LORA_DROPOUT = 0.1         # dropout nas camadas LoRA
    LORA_ENCODER_LR = 1e-5    # LR diferenciada para LoRA (10x menor que decoder)

    # =========================================================================
    # INFERENCIA AVANCADA
    # =========================================================================
    # TTA (Test-Time Augmentation): media de predicoes com flips/rotacao
    # Melhora IoU em +2-4% ao custo de 4x mais tempo de inferencia
    USE_TTA = True
    TTA_TRANSFORMS = ["original", "hflip", "vflip", "rot180"]

    # Refinamento 2-pass: usa mascara do 1o pass como prompt para o 2o
    USE_MASK_REFINEMENT = True

    # Filtro de slope (DEM) para lakes: rejeita deteccoes em areas ingremes
    # Lagos nao existem em slopes > threshold (a agua escorre)
    USE_SLOPE_FILTER = True
    SLOPE_FILTER_MAX_DEGREES = 15.0  # graus

    # =========================================================================
    # DETECÇÃO DE SOMBRA (DEM-based)
    # =========================================================================
    # Ângulos solares típicos para Tierra del Fuego (~54°S, verão austral)
    # Sol sempre ao norte no hemisfério sul
    SHADOW_SOLAR_AZIMUTHS = [330, 0, 30]    # graus (0=Norte, sentido horário)
    SHADOW_SOLAR_ALTITUDES = [30, 40, 50]   # graus acima do horizonte
    SHADOW_HILLSHADE_THRESHOLD = 80         # hillshade < threshold = sombra
    SHADOW_TEXTURE_MIN_VARIANCE = 15.0      # variância mínima para manter componente
    
    # =========================================================================
    # MÉTODOS
    # =========================================================================
    @classmethod
    def create_directories(cls):
        """Cria estrutura de diretórios do projeto."""
        dirs_to_create = [
            cls.DATA_DIR,
            cls.TILES_DIR,
            cls.MASKS_DIR,
            cls.MODELS_DIR,
            cls.RESULTS_DIR
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Criar subdiretórios por ano
        for year in cls.YEARS:
            (cls.TILES_DIR / str(year)).mkdir(exist_ok=True)
            (cls.MASKS_DIR / str(year)).mkdir(exist_ok=True)
            (cls.MASKS_DIR / str(year) / "annotations").mkdir(exist_ok=True)
            (cls.RESULTS_DIR / str(year)).mkdir(exist_ok=True)
            
            # Subdiretórios por feição
            for feature in cls.FEATURES.keys():
                (cls.MASKS_DIR / str(year) / feature).mkdir(exist_ok=True)
        
        print(f"✓ Estrutura de diretórios criada em {cls.PROJECT_DIR}")
    
    @classmethod
    def get_mosaic_path(cls, year: int) -> Path:
        """Retorna caminho para mosaico de um ano específico."""
        if year not in cls.MOSAICS:
            raise ValueError(f"Ano {year} não disponível. Anos válidos: {cls.YEARS}")
        preferred = cls.DATA_SOURCE_DIR / cls.MOSAICS[year]
        if preferred.exists():
            return preferred
        return cls._resolve_data_file(year=year, kind="mosaic", fallback=preferred)
    
    @classmethod
    def get_dem_path(cls, year: int) -> Path:
        """Retorna caminho para DEM de um ano específico."""
        if year not in cls.DEMS:
            raise ValueError(f"DEM do ano {year} não disponível.")
        preferred = cls.DATA_SOURCE_DIR / cls.DEMS[year]
        if preferred.exists():
            return preferred
        return cls._resolve_data_file(year=year, kind="dem", fallback=preferred)

    @classmethod
    def _resolve_data_file(cls, year: int, kind: str, fallback: Path) -> Path:
        """
        Resolve automaticamente nomes de arquivo com sufixos
        (ex: Schiaparelli_mosaic_2018-006.tif).
        """
        patterns = [
            f"*{year}*.tif",
            f"*{year}*.TIF",
            f"*{year}*.tiff",
            f"*{year}*.TIFF",
        ]
        candidates = []
        for pattern in patterns:
            candidates.extend([p for p in cls.DATA_SOURCE_DIR.glob(pattern) if p.is_file()])

        kind_l = kind.lower()
        filtered = [p for p in candidates if kind_l in p.name.lower()]
        if not filtered:
            return fallback
        return sorted(filtered, key=lambda p: p.name)[0]
    
    @classmethod
    def print_info(cls):
        """Imprime informações sobre a configuração atual."""
        print("=" * 60)
        print("CONFIGURAÇÃO DO PROJETO - SAM GLACIAR SCHIAPARELLI")
        print("=" * 60)
        print(f"Diretório do projeto: {cls.PROJECT_DIR}")
        print(f"Pasta de dados brutos: {cls.DATA_SOURCE_DIR}")
        print(f"Modelo SAM: {cls.MODEL_TYPE} ({'HQ' if cls.USE_SAM_HQ else 'standard'})")
        print(f"Checkpoint: {cls.SAM_CHECKPOINT.name}")
        print(f"Device: {cls.DEVICE}")
        print(f"Tile size: {cls.TILE_SIZE}x{cls.TILE_SIZE} (overlap: {cls.OVERLAP})")
        print(f"Anos disponíveis: {cls.YEARS}")
        print(f"Feições alvo: {list(cls.FEATURES.keys())}")
        print(f"Augmentation: {'SIM' if cls.USE_AUGMENTATION else 'NAO'}")
        print(f"LoRA: {'SIM (r={}, alpha={})'.format(cls.LORA_RANK, cls.LORA_ALPHA) if cls.USE_LORA else 'NAO'}")
        print("=" * 60)


# Executar ao importar para verificar configuração
if __name__ == "__main__":
    Config.print_info()
    Config.create_directories()
