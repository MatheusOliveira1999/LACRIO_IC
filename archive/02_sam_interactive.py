"""
02_sam_interactive.py - Interface de anotação interativa com SAM

Projeto: LACRIO IC - Extração de Feições Supraglaciais
Data: Janeiro 2026

Uso:
    python 02_sam_interactive.py                    # Inicia com ano 2016
    python 02_sam_interactive.py --year 2019       # Inicia com ano específico
    python 02_sam_interactive.py --tile 500        # Inicia em tile específico

Controles:
    Clique esquerdo  - Prompt positivo (incluir região)
    Clique direito   - Prompt negativo (excluir região)
    ←/→              - Navegar entre tiles
    1/2/3            - Selecionar feição (lake/crevasse/channel)
    Enter            - Salvar máscara atual
    R                - Resetar prompts
    S                - Saltar 10 tiles
    A                - Voltar 10 tiles
    M                - Alternar modo multi-mask
    Q                - Sair e salvar progresso
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch

# Importar configurações
sys.path.insert(0, str(Path(__file__).parent))
from config import Config

# Importar SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️  segment_anything não encontrado. Instalando...")
    print("   pip install git+https://github.com/facebookresearch/segment-anything.git")


class SAMAnnotator:
    """Interface de anotação interativa com SAM."""
    
    def __init__(self, year: int, start_tile: int = 0):
        """
        Inicializa o anotador SAM.
        
        Args:
            year: Ano dos tiles a anotar
            start_tile: Índice inicial do tile
        """
        self.year = year
        self.tiles_dir = Config.TILES_DIR / str(year)
        self.masks_dir = Config.MASKS_DIR / str(year) / "annotations"
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Carregar índice de tiles
        self.tiles_index = self._load_tiles_index()
        self.tile_ids = sorted([t["id"] for t in self.tiles_index])
        self.current_idx = self._find_tile_index(start_tile)
        
        # Estado da anotação
        self.current_feature = "lakes"
        self.features = list(Config.FEATURES.keys())
        self.prompt_points = []
        self.prompt_labels = []  # 1 = positivo, 0 = negativo
        self.multi_mask_mode = False
        self.current_mask = None
        self.current_mask_idx = 0
        
        # Carregar modelo SAM
        print(f"🔄 Carregando modelo SAM ({Config.MODEL_TYPE})...")
        self.sam = self._load_sam()
        self.predictor = SamPredictor(self.sam)
        print(f"✓ Modelo carregado em {Config.DEVICE}")
        
        # Carregar progresso anterior (ANTES de carregar tile)
        self.annotations = self._load_annotations_index()
        
        # Carregar imagem atual
        self.current_image = None
        self.display_image = None
        self._load_current_tile()
        
        # Window setup
        self.window_name = f"SAM Annotator - {year}"
        
    def _load_sam(self):
        """Carrega o modelo SAM."""
        if not SAM_AVAILABLE:
            raise RuntimeError("segment_anything não instalado")
        
        sam = sam_model_registry[Config.MODEL_TYPE](checkpoint=str(Config.SAM_CHECKPOINT))
        sam.to(device=Config.DEVICE)
        return sam
    
    def _load_tiles_index(self) -> list:
        """Carrega índice de tiles."""
        index_path = self.tiles_dir / "tiles_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Índice não encontrado: {index_path}")
        
        with open(index_path) as f:
            data = json.load(f)
        return data["tiles"]
    
    def _find_tile_index(self, tile_id: int) -> int:
        """Encontra o índice do tile na lista."""
        for i, tid in enumerate(self.tile_ids):
            if tid >= tile_id:
                return i
        return 0
    
    def _load_current_tile(self):
        """Carrega o tile atual."""
        tile_id = self.tile_ids[self.current_idx]
        tile_path = self.tiles_dir / f"tile_{tile_id:06d}.png"
        
        if not tile_path.exists():
            print(f"⚠️  Tile não encontrado: {tile_path}")
            return
        
        # Carregar imagem BGR (OpenCV padrão)
        self.current_image = cv2.imread(str(tile_path))
        
        # Converter para RGB para o SAM
        image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image_rgb)
        
        # Resetar estado
        self.prompt_points = []
        self.prompt_labels = []
        self.current_mask = None
        self.current_mask_idx = 0
        
        self._update_display()
    
    def _update_display(self):
        """Atualiza a imagem de display."""
        if self.current_image is None:
            return
            
        self.display_image = self.current_image.copy()
        
        # Overlay da máscara se existir
        if self.current_mask is not None:
            color = Config.FEATURES[self.current_feature]["color"]
            overlay = self.display_image.copy()
            overlay[self.current_mask > 0] = color
            cv2.addWeighted(overlay, 0.4, self.display_image, 0.6, 0, self.display_image)
            
            # Contorno da máscara
            contours, _ = cv2.findContours(
                self.current_mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(self.display_image, contours, -1, color, 2)
        
        # Desenhar prompts
        for pt, label in zip(self.prompt_points, self.prompt_labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)  # Verde/Vermelho
            marker = cv2.MARKER_CROSS if label == 1 else cv2.MARKER_TILTED_CROSS
            cv2.drawMarker(self.display_image, pt, color, marker, 15, 2)
        
        # Info bar
        self._draw_info_bar()
    
    def _draw_info_bar(self):
        """Desenha barra de informações."""
        h, w = self.display_image.shape[:2]
        bar_height = 40
        
        # Fundo da barra
        cv2.rectangle(self.display_image, (0, h - bar_height), (w, h), (40, 40, 40), -1)
        
        # Informações
        tile_id = self.tile_ids[self.current_idx]
        annotated = tile_id in self.annotations.get(self.current_feature, set())
        status = "✓" if annotated else "○"
        
        info_text = (
            f"Tile: {tile_id:06d}/{len(self.tile_ids)-1} | "
            f"Feature: {self.current_feature.upper()} | "
            f"Prompts: {len(self.prompt_points)} | "
            f"Multi: {'ON' if self.multi_mask_mode else 'OFF'} | "
            f"Status: {status}"
        )
        
        cv2.putText(
            self.display_image, info_text, (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        # Teclas de ajuda
        help_text = "[1/2/3] Feature | [Enter] Save | [R] Reset | [Q] Quit"
        cv2.putText(
            self.display_image, help_text, (w - 400, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1
        )
    
    def _segment(self):
        """Executa segmentação com SAM."""
        if not self.prompt_points:
            self.current_mask = None
            return
        
        input_points = np.array(self.prompt_points)
        input_labels = np.array(self.prompt_labels)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=self.multi_mask_mode
        )
        
        if self.multi_mask_mode:
            # Ciclar entre as máscaras
            self.current_mask_idx = self.current_mask_idx % len(masks)
            self.current_mask = masks[self.current_mask_idx]
        else:
            # Usar máscara com maior score
            best_idx = np.argmax(scores)
            self.current_mask = masks[best_idx]
        
        self._update_display()
    
    def _save_current_mask(self):
        """Salva a máscara atual."""
        if self.current_mask is None:
            print("⚠️  Nenhuma máscara para salvar")
            return False
        
        tile_id = self.tile_ids[self.current_idx]
        
        # Criar subdiretório da feição se não existir
        feature_dir = self.masks_dir / self.current_feature
        feature_dir.mkdir(exist_ok=True)
        
        # Salvar máscara como PNG (0 = fundo, 255 = feição)
        mask_filename = f"tile_{tile_id:06d}_{self.current_feature}.png"
        mask_path = feature_dir / mask_filename
        
        mask_uint8 = (self.current_mask * 255).astype(np.uint8)
        cv2.imwrite(str(mask_path), mask_uint8)
        
        # Atualizar índice
        if self.current_feature not in self.annotations:
            self.annotations[self.current_feature] = set()
        self.annotations[self.current_feature].add(tile_id)
        
        self._save_annotations_index()
        
        print(f"✓ Salvo: {mask_path.name}")
        return True
    
    def _load_annotations_index(self) -> dict:
        """Carrega índice de anotações."""
        index_path = self.masks_dir / "annotations_index.json"
        if not index_path.exists():
            return {}
        
        with open(index_path) as f:
            data = json.load(f)
        
        # Converter listas para sets
        return {k: set(v) for k, v in data.items()}
    
    def _save_annotations_index(self):
        """Salva índice de anotações."""
        index_path = self.masks_dir / "annotations_index.json"
        
        # Converter sets para listas para JSON
        data = {k: sorted(list(v)) for k, v in self.annotations.items()}
        data["_metadata"] = {
            "year": self.year,
            "last_updated": datetime.now().isoformat(),
            "total_annotations": sum(len(v) for v in self.annotations.values())
        }
        
        with open(index_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def on_mouse(self, event, x, y, flags, param):
        """Callback para eventos de mouse."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Prompt positivo
            self.prompt_points.append((x, y))
            self.prompt_labels.append(1)
            self._segment()
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Prompt negativo
            self.prompt_points.append((x, y))
            self.prompt_labels.append(0)
            self._segment()
    
    def run(self):
        """Loop principal da interface."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        
        print("\n" + "=" * 60)
        print("SAM ANNOTATOR - Interface de Anotação Interativa")
        print("=" * 60)
        print(f"Ano: {self.year}")
        print(f"Tiles: {len(self.tile_ids)}")
        print(f"Device: {Config.DEVICE}")
        print("\nControles:")
        print("  Clique esquerdo  - Prompt positivo (+)")
        print("  Clique direito   - Prompt negativo (-)")
        print("  ←/→              - Navegar tiles")
        print("  1/2/3            - Selecionar feição")
        print("  Enter            - Salvar máscara")
        print("  R                - Resetar prompts")
        print("  M                - Alternar multi-mask")
        print("  S/A              - Saltar ±10 tiles")
        print("  Q                - Sair")
        print("=" * 60 + "\n")
        
        while True:
            if self.display_image is not None:
                cv2.imshow(self.window_name, self.display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Sair
            if key == ord('q'):
                break
            
            # Navegar tiles
            elif key == 81 or key == ord('h'):  # Seta esquerda ou H
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self._load_current_tile()
            
            elif key == 83 or key == ord('l'):  # Seta direita ou L
                if self.current_idx < len(self.tile_ids) - 1:
                    self.current_idx += 1
                    self._load_current_tile()
            
            # Saltar tiles
            elif key == ord('s'):
                self.current_idx = min(self.current_idx + 10, len(self.tile_ids) - 1)
                self._load_current_tile()
            
            elif key == ord('a'):
                self.current_idx = max(self.current_idx - 10, 0)
                self._load_current_tile()
            
            # Selecionar feição
            elif key == ord('1'):
                self.current_feature = "lakes"
                self._update_display()
                print(f"🔵 Feição: LAKES")
            
            elif key == ord('2'):
                self.current_feature = "crevasses"
                self._update_display()
                print(f"🔴 Feição: CREVASSES")
            
            elif key == ord('3'):
                self.current_feature = "channels"
                self._update_display()
                print(f"🔵 Feição: CHANNELS")
            
            # Salvar máscara
            elif key == 13:  # Enter
                if self._save_current_mask():
                    # Avançar para próximo tile
                    if self.current_idx < len(self.tile_ids) - 1:
                        self.current_idx += 1
                        self._load_current_tile()
            
            # Resetar prompts
            elif key == ord('r'):
                self.prompt_points = []
                self.prompt_labels = []
                self.current_mask = None
                self._update_display()
                print("🔄 Prompts resetados")
            
            # Alternar multi-mask
            elif key == ord('m'):
                self.multi_mask_mode = not self.multi_mask_mode
                if self.prompt_points:
                    self._segment()
                else:
                    self._update_display()
                print(f"🔀 Multi-mask: {'ON' if self.multi_mask_mode else 'OFF'}")
            
            # Ciclar máscaras (quando multi-mask ativo)
            elif key == ord('n') and self.multi_mask_mode and self.current_mask is not None:
                self.current_mask_idx += 1
                self._segment()
        
        cv2.destroyAllWindows()
        self._save_annotations_index()
        
        # Resumo final
        total = sum(len(v) for v in self.annotations.values())
        print(f"\n✅ Sessão encerrada")
        print(f"   Total de anotações: {total}")
        for feature, tiles in self.annotations.items():
            if feature != "_metadata":
                print(f"   - {feature}: {len(tiles)} tiles")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Interface de anotação interativa com SAM"
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        default=2016,
        choices=Config.YEARS,
        help=f"Ano dos tiles (default: 2016)"
    )
    parser.add_argument(
        "--tile", "-t",
        type=int,
        default=0,
        help="Tile inicial (default: 0)"
    )
    
    args = parser.parse_args()
    
    if not SAM_AVAILABLE:
        print("❌ Erro: segment_anything não instalado")
        print("   Execute: pip install git+https://github.com/facebookresearch/segment-anything.git")
        sys.exit(1)
    
    if not Config.SAM_CHECKPOINT.exists():
        print(f"❌ Erro: Modelo SAM não encontrado: {Config.SAM_CHECKPOINT}")
        sys.exit(1)
    
    Config.print_info()
    
    annotator = SAMAnnotator(args.year, args.tile)
    annotator.run()


if __name__ == "__main__":
    main()
