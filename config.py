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

class Config:
    """Configurações centralizadas do projeto."""
    
    # =========================================================================
    # DIRETÓRIOS
    # =========================================================================
    PROJECT_DIR = Path("/home/matheus/Documents/GitHub/LACRIO IC")
    DATA_SOURCE_DIR = PROJECT_DIR / "mosaicos_DEMs_Schiaparelli"  # Onde estão os mosaicos .tif
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
    SAM_CHECKPOINT = PROJECT_DIR / "sam_vit_b_01ec64.pth"
    
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
            "min_area": 50,            # Área mínima em pixels
            "max_area": 50000          # Área máxima em pixels
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
    # TREINAMENTO
    # =========================================================================
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    TRAIN_VAL_SPLIT = 0.8
    
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
        return cls.DATA_SOURCE_DIR / cls.MOSAICS[year]
    
    @classmethod
    def get_dem_path(cls, year: int) -> Path:
        """Retorna caminho para DEM de um ano específico."""
        if year not in cls.DEMS:
            raise ValueError(f"DEM do ano {year} não disponível.")
        return cls.DATA_SOURCE_DIR / cls.DEMS[year]
    
    @classmethod
    def print_info(cls):
        """Imprime informações sobre a configuração atual."""
        print("=" * 60)
        print("CONFIGURAÇÃO DO PROJETO - SAM GLACIAR SCHIAPARELLI")
        print("=" * 60)
        print(f"Diretório do projeto: {cls.PROJECT_DIR}")
        print(f"Modelo SAM: {cls.MODEL_TYPE}")
        print(f"Device: {cls.DEVICE}")
        print(f"Tile size: {cls.TILE_SIZE}x{cls.TILE_SIZE} (overlap: {cls.OVERLAP})")
        print(f"Anos disponíveis: {cls.YEARS}")
        print(f"Feições alvo: {list(cls.FEATURES.keys())}")
        print("=" * 60)


# Executar ao importar para verificar configuração
if __name__ == "__main__":
    Config.print_info()
    Config.create_directories()
