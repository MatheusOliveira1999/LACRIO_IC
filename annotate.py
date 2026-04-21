"""
annotate.py - Ferramenta de anotacao interativa para tiles supraglaciais

Uso:
    python annotate.py                          # usa active_learning_lakes.csv
    python annotate.py --feature lakes --year 2017
    python annotate.py --csv results/active_learning_lakes.csv
    python annotate.py --review --year 2016     # revisar tiles JA anotados

Controles:
    Pincel esquerdo     → adicionar lago (pintar vermelho)
    Pincel direito      → apagar (borracha)
    A                   → aceitar predicao do modelo como anotacao
    R                   → rejeitar (salvar mascara vazia = sem lago)
    Z                   → desfazer ultima pincelada
    Enter / Espaco      → salvar e proximo tile
    S                   → pular (nao salvar, ir ao proximo)
    Q / Esc             → sair
    Scroll              → aumentar/diminuir tamanho do pincel
    Ctrl+Scroll         → zoom in/out (centralizado no cursor)
    Botao meio + arrastar → mover visao (pan)
    0                   → resetar zoom
"""

import tkinter as tk
from tkinter import ttk, messagebox
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
import torch

from config import Config
import importlib
_unet = importlib.import_module("03b_train_unet")
_inf  = importlib.import_module("04b_inference_unet")
UNetResNet34 = _unet.UNetResNet34
load_unet    = _inf.load_unet
preprocess_image = _inf.preprocess_image


# ============================================================================
# Carregar lista de tiles
# ============================================================================

def load_tile_list(feature, year=None, csv_path=None, review=False):
    """Carrega lista de tiles para anotar.

    review=False: lista tiles ainda NAO anotados (fluxo normal).
    review=True:  lista tiles JA anotados (para revisar/corrigir anotacoes).
    """

    tiles = []

    if review:
        # Modo revisao: coletar tiles que ja tem anotacao salva
        years = [year] if year else Config.YEARS
        for yr in years:
            gt_dir = Config.MASKS_DIR / str(yr) / "annotations" / feature
            if not gt_dir.exists():
                continue
            for gt_path in sorted(gt_dir.glob(f"tile_*_{feature}.png")):
                tid = gt_path.stem.replace(f"_{feature}", "")
                tile_path = Config.TILES_DIR / str(yr) / f"{tid}.png"
                if tile_path.exists():
                    tiles.append({
                        "tile_id": tid,
                        "year": yr,
                        "tile_path": str(tile_path),
                        "score": 0,
                    })
        return tiles

    if csv_path and Path(csv_path).exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if year and int(row["year"]) != year:
                    continue
                tiles.append({
                    "tile_id": row["tile_id"],
                    "year": int(row["year"]),
                    "tile_path": row["tile_path"],
                    "score": float(row.get("uncertainty_score", 0)),
                })
    else:
        # Fallback: carregar todos os tiles de um ano sem anotacao
        years = [year] if year else Config.YEARS
        for yr in years:
            gt_dir = Config.MASKS_DIR / str(yr) / "annotations" / feature
            annotated = set()
            if gt_dir.exists():
                for p in gt_dir.glob(f"tile_*_{feature}.png"):
                    annotated.add(p.stem.replace(f"_{feature}", ""))

            idx_path = Config.TILES_DIR / str(yr) / "tiles_index.json"
            if not idx_path.exists():
                continue
            with open(idx_path) as f:
                index = json.load(f)["tiles"]
            for t in index:
                tid = f"tile_{t['id']:06d}"
                if tid not in annotated:
                    tp = Config.TILES_DIR / str(yr) / t["filename"]
                    if tp.exists():
                        tiles.append({"tile_id": tid, "year": yr,
                                      "tile_path": str(tp), "score": 0})

    # Excluir ja anotados
    result = []
    for t in tiles:
        gt_path = (Config.MASKS_DIR / str(t["year"]) / "annotations" / feature
                   / f"{t['tile_id']}_{feature}.png")
        if not gt_path.exists():
            result.append(t)

    return result


def get_model_prediction(model, img_size, tile_path):
    """Roda U-Net e retorna mapa de probabilidade."""
    image = cv2.imread(str(tile_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    tensor = preprocess_image(image, img_size).to(Config.DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits[0, 0]).cpu().numpy()
    prob_full = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    pred_mask = (prob_full > 0.5).astype(np.uint8) * 255
    return image, pred_mask


# ============================================================================
# Aplicacao de anotacao
# ============================================================================

class AnnotationApp:
    DISPLAY_SIZE = 512   # tamanho de exibicao de cada painel
    BRUSH_DEFAULT = 10

    def __init__(self, root, tiles, feature, model, img_size, review=False):
        self.root = root
        self.tiles = tiles
        self.feature = feature
        self.model = model
        self.img_size = img_size
        self.review = review

        self.idx = 0
        self.brush_size = self.BRUSH_DEFAULT
        self.drawing = False
        self.erasing = False
        self.undo_stack = []

        # Zoom / pan state (em coordenadas do display S×S)
        self.zoom_level = 1.0   # 1.0 = visao completa
        self.view_x = 0.0       # canto superior esquerdo da visao (S-space)
        self.view_y = 0.0
        self._pan_start = None  # (canvas_x, canvas_y) ao pressionar botao meio
        self._pan_view0 = None  # (view_x, view_y) ao iniciar pan

        self.image = None       # RGB np array (original)
        self.mask = None        # uint8 np array (anotacao atual)
        self.pred_mask = None   # uint8 np array (predicao do modelo)

        self._build_ui()
        self._load_tile()

    def _build_ui(self):
        mode_str = " [MODO REVISAO]" if self.review else ""
        self.root.title(f"Ferramenta de Anotacao - LACRIO IC{mode_str}")
        self.root.resizable(False, False)

        # Barra de status no topo
        top = tk.Frame(self.root, bg="#1e1e1e", pady=6)
        top.pack(fill=tk.X)

        self.lbl_progress = tk.Label(top, text="", fg="white", bg="#1e1e1e",
                                     font=("Courier", 11, "bold"))
        self.lbl_progress.pack(side=tk.LEFT, padx=12)

        self.lbl_tile = tk.Label(top, text="", fg="#aaaaaa", bg="#1e1e1e",
                                 font=("Courier", 10))
        self.lbl_tile.pack(side=tk.LEFT, padx=8)

        self.lbl_brush = tk.Label(top, text="", fg="#ffcc00", bg="#1e1e1e",
                                  font=("Courier", 10))
        self.lbl_brush.pack(side=tk.RIGHT, padx=12)

        # Paineis de imagem
        panels = tk.Frame(self.root, bg="#111")
        panels.pack()

        S = self.DISPLAY_SIZE

        # Esquerdo: imagem original
        lf1 = tk.LabelFrame(panels, text=" Original ", fg="white", bg="#111",
                             font=("Courier", 9))
        lf1.grid(row=0, column=0, padx=4, pady=4)
        self.canvas_orig = tk.Canvas(lf1, width=S, height=S, bg="black",
                                     cursor="crosshair")
        self.canvas_orig.pack()

        # Direito: anotacao (editavel)
        lf2 = tk.LabelFrame(panels, text=" Anotacao (editar aqui) ",
                             fg="#ff6666", bg="#111", font=("Courier", 9))
        lf2.grid(row=0, column=1, padx=4, pady=4)
        self.canvas_ann = tk.Canvas(lf2, width=S, height=S, bg="black",
                                    cursor="crosshair")
        self.canvas_ann.pack()

        # Mouse events no canvas de anotacao
        self.canvas_ann.bind("<ButtonPress-1>",    self._on_press_left)
        self.canvas_ann.bind("<B1-Motion>",         self._on_drag_left)
        self.canvas_ann.bind("<ButtonRelease-1>",   self._on_release)
        self.canvas_ann.bind("<ButtonPress-3>",     self._on_press_right)
        self.canvas_ann.bind("<B3-Motion>",         self._on_drag_right)
        self.canvas_ann.bind("<ButtonRelease-3>",   self._on_release)
        self.canvas_ann.bind("<MouseWheel>",        self._on_scroll)
        self.canvas_ann.bind("<Button-4>",          self._on_scroll)
        self.canvas_ann.bind("<Button-5>",          self._on_scroll)
        # Zoom com Ctrl+Scroll (Linux: Button-4/5; Windows/Mac: MouseWheel com Ctrl)
        self.canvas_ann.bind("<Control-MouseWheel>", self._on_zoom_scroll)
        self.canvas_ann.bind("<Control-Button-4>",   self._on_zoom_scroll)
        self.canvas_ann.bind("<Control-Button-5>",   self._on_zoom_scroll)
        # Pan com botao do meio
        self.canvas_ann.bind("<ButtonPress-2>",    self._on_pan_press)
        self.canvas_ann.bind("<B2-Motion>",         self._on_pan_drag)
        self.canvas_ann.bind("<ButtonRelease-2>",   self._on_pan_release)
        # Pan tambem no canvas original (so visualizacao)
        self.canvas_orig.bind("<Control-MouseWheel>", self._on_zoom_scroll)
        self.canvas_orig.bind("<Control-Button-4>",   self._on_zoom_scroll)
        self.canvas_orig.bind("<Control-Button-5>",   self._on_zoom_scroll)
        self.canvas_orig.bind("<ButtonPress-2>",    self._on_pan_press)
        self.canvas_orig.bind("<B2-Motion>",         self._on_pan_drag)
        self.canvas_orig.bind("<ButtonRelease-2>",   self._on_pan_release)

        # Barra de botoes
        btn_frame = tk.Frame(self.root, bg="#1e1e1e", pady=8)
        btn_frame.pack(fill=tk.X)

        btn_cfg = {"font": ("Courier", 11, "bold"), "width": 14, "height": 1,
                   "relief": tk.FLAT, "bd": 0, "padx": 6}

        tk.Button(btn_frame, text="[A] Aceitar pred",
                  bg="#2255aa", fg="white", command=self._accept_pred,
                  **btn_cfg).pack(side=tk.LEFT, padx=6)

        tk.Button(btn_frame, text="[R] Rejeitar",
                  bg="#884400", fg="white", command=self._reject,
                  **btn_cfg).pack(side=tk.LEFT, padx=6)

        tk.Button(btn_frame, text="[Z] Desfazer",
                  bg="#444444", fg="white", command=self._undo,
                  **btn_cfg).pack(side=tk.LEFT, padx=6)

        tk.Button(btn_frame, text="[S] Pular",
                  bg="#333333", fg="#aaaaaa", command=self._skip,
                  **btn_cfg).pack(side=tk.RIGHT, padx=6)

        tk.Button(btn_frame, text="[Enter] Salvar →",
                  bg="#226622", fg="white", command=self._save_and_next,
                  **btn_cfg).pack(side=tk.RIGHT, padx=6)

        if self.review:
            tk.Button(btn_frame, text="[D] Apagar GT",
                      bg="#aa2222", fg="white", command=self._delete_annotation,
                      **btn_cfg).pack(side=tk.LEFT, padx=6)

        # Instrucoes
        info = tk.Label(self.root,
                        text="Esq: pintar  |  Dir: apagar  |  Scroll: pincel  |  "
                             "Ctrl+Scroll: zoom  |  BotaoMeio: pan  |  0: reset zoom",
                        fg="#888888", bg="#111", font=("Courier", 9))
        info.pack(pady=4)

        # Atalhos de teclado
        self.root.bind("<Return>",  lambda e: self._save_and_next())
        self.root.bind("<space>",   lambda e: self._save_and_next())
        self.root.bind("a",         lambda e: self._accept_pred())
        self.root.bind("A",         lambda e: self._accept_pred())
        self.root.bind("r",         lambda e: self._reject())
        self.root.bind("R",         lambda e: self._reject())
        self.root.bind("s",         lambda e: self._skip())
        self.root.bind("S",         lambda e: self._skip())
        self.root.bind("z",         lambda e: self._undo())
        self.root.bind("Z",         lambda e: self._undo())
        self.root.bind("<Escape>",  lambda e: self._quit())
        self.root.bind("q",         lambda e: self._quit())
        self.root.bind("Q",         lambda e: self._quit())
        self.root.bind("+",         lambda e: self._change_brush(+3))
        self.root.bind("-",         lambda e: self._change_brush(-3))
        self.root.bind("0",         lambda e: self._reset_zoom())
        self.root.bind("d",         lambda e: self._delete_annotation())
        self.root.bind("D",         lambda e: self._delete_annotation())

    def _load_tile(self):
        if self.idx >= len(self.tiles):
            messagebox.showinfo("Concluido",
                                f"Todos os {len(self.tiles)} tiles foram anotados!")
            self.root.quit()
            return

        t = self.tiles[self.idx]
        self.image, self.pred_mask = get_model_prediction(
            self.model, self.img_size, t["tile_path"]
        )

        # No modo revisao: carregar mascara existente como ponto de partida
        gt_path = (Config.MASKS_DIR / str(t["year"]) / "annotations"
                   / self.feature / f"{t['tile_id']}_{self.feature}.png")
        if self.review and gt_path.exists():
            existing = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if existing is not None and existing.shape == self.image.shape[:2]:
                self.mask = existing.copy()
            else:
                self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        else:
            self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.undo_stack = []
        # Resetar zoom ao mudar de tile
        self.zoom_level = 1.0
        self.view_x = 0.0
        self.view_y = 0.0

        n = len(self.tiles)
        self.lbl_progress.config(text=f"Tile {self.idx+1}/{n}")
        self.lbl_tile.config(text=f"{t['tile_id']}  ano={t['year']}  "
                                   f"score={t.get('score',0):.0f}")
        self._update_brush_label()
        self._render()

    def _get_zoomed_view(self, arr_s):
        """Recorta e redimensiona arr_s (S×S ou S×S×3) de acordo com zoom/pan."""
        S = self.DISPLAY_SIZE
        if self.zoom_level <= 1.0:
            return arr_s
        visible = S / self.zoom_level       # largura visivel em pixels S-space
        x0 = int(self.view_x)
        y0 = int(self.view_y)
        x1 = max(x0 + 1, min(S, int(x0 + visible)))
        y1 = max(y0 + 1, min(S, int(y0 + visible)))
        cropped = arr_s[y0:y1, x0:x1]
        if cropped.size == 0:
            return arr_s
        interp = cv2.INTER_NEAREST if arr_s.ndim == 2 else cv2.INTER_LINEAR
        return cv2.resize(cropped, (S, S), interpolation=interp)

    def _clamp_view(self):
        """Mantém view dentro dos limites validos."""
        S = self.DISPLAY_SIZE
        visible = S / self.zoom_level
        max_v = S - visible
        self.view_x = max(0.0, min(self.view_x, max_v))
        self.view_y = max(0.0, min(self.view_y, max_v))

    def _render(self):
        S = self.DISPLAY_SIZE

        # Painel esquerdo: original
        orig_resized = cv2.resize(self.image, (S, S))
        self._draw_canvas(self.canvas_orig, self._get_zoomed_view(orig_resized))

        # Painel direito: anotacao atual com overlay vermelho
        ann_resized = cv2.resize(self.image, (S, S))
        mask_resized = cv2.resize(self.mask, (S, S),
                                  interpolation=cv2.INTER_NEAREST)
        pred_resized = cv2.resize(self.pred_mask, (S, S),
                                  interpolation=cv2.INTER_NEAREST)

        # Predicao em azul claro (fundo de referencia)
        pred_pixels = pred_resized > 127
        ann_resized[pred_pixels] = (
            ann_resized[pred_pixels] * 0.6 +
            np.array([100, 160, 255]) * 0.4
        ).astype(np.uint8)

        # Anotacao atual em vermelho opaco
        ann_pixels = mask_resized > 127
        ann_resized[ann_pixels] = (
            ann_resized[ann_pixels] * 0.4 +
            np.array([255, 60, 60]) * 0.6
        ).astype(np.uint8)

        self._draw_canvas(self.canvas_ann, self._get_zoomed_view(ann_resized))

    def _draw_canvas(self, canvas, rgb_array):
        img = Image.fromarray(rgb_array)
        photo = ImageTk.PhotoImage(img)
        canvas.photo = photo  # manter referencia
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    # ---- Mouse ----

    def _canvas_to_mask(self, cx, cy):
        """Converte coordenadas do canvas (display) para coordenadas da mascara."""
        S = self.DISPLAY_SIZE
        h, w = self.mask.shape
        # Canvas -> S-space (accounting for zoom and pan)
        img_sx = self.view_x + cx / self.zoom_level
        img_sy = self.view_y + cy / self.zoom_level
        # S-space -> mask space
        mx = int(img_sx * w / S)
        my = int(img_sy * h / S)
        return np.clip(mx, 0, w-1), np.clip(my, 0, h-1)

    def _paint(self, cx, cy, value):
        mx, my = self._canvas_to_mask(cx, cy)
        S = self.DISPLAY_SIZE
        h, w = self.mask.shape
        # Brush size em pixels da mascara, escalado pelo zoom
        # (pincel sempre ocupa o mesmo numero de pixels no ecra)
        br = max(1, int(self.brush_size * w / S / self.zoom_level))
        y1 = max(0, my - br); y2 = min(h, my + br)
        x1 = max(0, mx - br); x2 = min(w, mx + br)
        self.mask[y1:y2, x1:x2] = value
        self._render()

    def _on_press_left(self, e):
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)
        self.drawing = True
        self._paint(e.x, e.y, 255)

    def _on_drag_left(self, e):
        if self.drawing:
            self._paint(e.x, e.y, 255)

    def _on_press_right(self, e):
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)
        self.erasing = True
        self._paint(e.x, e.y, 0)

    def _on_drag_right(self, e):
        if self.erasing:
            self._paint(e.x, e.y, 0)

    def _on_release(self, e):
        self.drawing = False
        self.erasing = False

    def _on_scroll(self, e):
        # Ignorar se Ctrl estiver pressionado (sera tratado por _on_zoom_scroll)
        if e.state & 0x4:
            return
        if hasattr(e, "delta") and e.delta:
            self._change_brush(3 if e.delta > 0 else -3)
        elif e.num == 4:
            self._change_brush(3)
        elif e.num == 5:
            self._change_brush(-3)

    def _on_zoom_scroll(self, e):
        """Ctrl+Scroll: zoom centrado na posicao do cursor."""
        S = self.DISPLAY_SIZE
        cx, cy = e.x, e.y

        # Determinar direcao
        if hasattr(e, "delta") and e.delta:
            zoom_in = e.delta > 0
        else:
            zoom_in = (e.num == 4)

        # Posicao do cursor em S-space antes do zoom
        img_sx = self.view_x + cx / self.zoom_level
        img_sy = self.view_y + cy / self.zoom_level

        # Novo zoom (limite entre 1x e 64x)
        factor = 1.4 if zoom_in else (1.0 / 1.4)
        new_zoom = max(1.0, min(64.0, self.zoom_level * factor))
        if new_zoom == self.zoom_level:
            return
        self.zoom_level = new_zoom

        # Ajustar view para manter cursor na mesma posicao de imagem
        self.view_x = img_sx - cx / self.zoom_level
        self.view_y = img_sy - cy / self.zoom_level
        self._clamp_view()
        self._update_brush_label()
        self._render()

    def _on_pan_press(self, e):
        self._pan_start = (e.x, e.y)
        self._pan_view0 = (self.view_x, self.view_y)

    def _on_pan_drag(self, e):
        if self._pan_start is None:
            return
        dx = e.x - self._pan_start[0]
        dy = e.y - self._pan_start[1]
        self.view_x = self._pan_view0[0] - dx / self.zoom_level
        self.view_y = self._pan_view0[1] - dy / self.zoom_level
        self._clamp_view()
        self._render()

    def _on_pan_release(self, e):
        self._pan_start = None
        self._pan_view0 = None

    def _reset_zoom(self):
        self.zoom_level = 1.0
        self.view_x = 0.0
        self.view_y = 0.0
        self._render()

    def _change_brush(self, delta):
        self.brush_size = max(2, min(80, self.brush_size + delta))
        self._update_brush_label()

    def _update_brush_label(self):
        zoom_str = f"  zoom:{self.zoom_level:.1f}x" if self.zoom_level > 1.0 else ""
        self.lbl_brush.config(text=f"Pincel: {self.brush_size}px{zoom_str}")

    # ---- Acoes ----

    def _accept_pred(self):
        """Copia predicao do modelo para a anotacao."""
        self.undo_stack.append(self.mask.copy())
        self.mask = self.pred_mask.copy()
        self._render()

    def _reject(self):
        """Limpa a mascara (sem lago)."""
        self.undo_stack.append(self.mask.copy())
        self.mask = np.zeros_like(self.mask)
        self._render()

    def _undo(self):
        if self.undo_stack:
            self.mask = self.undo_stack.pop()
            self._render()

    def _save_and_next(self):
        """Salva a mascara atual e vai para o proximo tile."""
        t = self.tiles[self.idx]
        out_dir = (Config.MASKS_DIR / str(t["year"]) / "annotations"
                   / self.feature)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{t['tile_id']}_{self.feature}.png"
        cv2.imwrite(str(out_path), self.mask)
        n_pixels = (self.mask > 127).sum()
        print(f"  Salvo: {out_path.name}  ({n_pixels} pixels)")
        self.idx += 1
        self._load_tile()

    def _skip(self):
        """Pula sem salvar."""
        print(f"  Pulado: {self.tiles[self.idx]['tile_id']}")
        self.idx += 1
        self._load_tile()

    def _delete_annotation(self):
        """Remove a anotacao salva deste tile do disco e vai para o proximo."""
        t = self.tiles[self.idx]
        gt_path = (Config.MASKS_DIR / str(t["year"]) / "annotations"
                   / self.feature / f"{t['tile_id']}_{self.feature}.png")
        if not gt_path.exists():
            print(f"  (nada pra apagar: {t['tile_id']})")
            self.idx += 1
            self._load_tile()
            return
        if not messagebox.askyesno(
                "Apagar anotacao",
                f"Apagar a anotacao de {t['tile_id']} ({t['year']})?\n"
                f"Esta acao nao pode ser desfeita."):
            return
        gt_path.unlink()
        print(f"  APAGADO: {gt_path.name}")
        self.idx += 1
        self._load_tile()

    def _quit(self):
        if messagebox.askyesno("Sair", "Sair da ferramenta de anotacao?"):
            self.root.quit()


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ferramenta de anotacao interativa")
    parser.add_argument("--feature", type=str, default="lakes",
                        choices=list(Config.FEATURES.keys()))
    parser.add_argument("--year", type=int, default=None, choices=Config.YEARS)
    parser.add_argument("--csv", type=str,
                        default=None,
                        help="CSV do active learning (default: results/active_learning_FEATURE.csv)")
    parser.add_argument("--review", action="store_true",
                        help="Modo revisao: abre tiles JA anotados para corrigir/apagar")
    args = parser.parse_args()

    print("Carregando lista de tiles...")
    if args.csv is None and not args.review:
        args.csv = str(Config.RESULTS_DIR / f"active_learning_{args.feature}.csv")
    csv_arg = None if args.review else args.csv
    tiles = load_tile_list(args.feature, args.year, csv_arg, review=args.review)

    if not tiles:
        msg = "Nenhum tile para revisar." if args.review else "Nenhum tile para anotar."
        print(msg)
        return

    mode = "revisar" if args.review else "anotar"
    print(f"  {len(tiles)} tiles para {mode}")
    if args.year:
        print(f"  Ano: {args.year}")

    print("Carregando modelo U-Net...")
    model, img_size = load_unet(args.feature)
    model.eval()

    root = tk.Tk()
    root.configure(bg="#111")
    app = AnnotationApp(root, tiles, args.feature, model, img_size,
                        review=args.review)
    root.mainloop()

    print(f"\nSessao encerrada. Tiles anotados nesta sessao: {app.idx}")
    print(f"Para re-treinar: python 03b_train_unet.py --feature {args.feature}")


if __name__ == "__main__":
    main()
