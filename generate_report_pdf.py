"""
Gerador de relatório PDF — LACRIO IC
Projeto: Extração de Feições Supraglaciais no Glaciar Schiaparelli
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from datetime import date

OUTPUT_PATH = "/home/matheus/Documents/Github/LACRIO_IC/results/Relatorio_LACRIO_IC_2026.pdf"

# ── Largura útil da página ────────────────────────────────────────────────────
PW = A4[0] - 4 * cm   # 17 cm

# ── Paleta ───────────────────────────────────────────────────────────────────
AZUL      = colors.HexColor("#1B4F72")
AZUL_MED  = colors.HexColor("#2E86C1")
AZUL_PAL  = colors.HexColor("#D6EAF8")
CINZA     = colors.HexColor("#AEB6BF")
BRANCO    = colors.white

# ── Estilos de parágrafo ─────────────────────────────────────────────────────
_base = getSampleStyleSheet()

def _S(nome, pai="Normal", **kw):
    return ParagraphStyle(nome, parent=_base[pai], **kw)

TITLE    = _S("TITLE",   "Title",   fontSize=22, textColor=AZUL, spaceAfter=6, leading=28)
SUBTITLE = _S("SUBTITLE","Normal",  fontSize=13, textColor=AZUL_MED, spaceAfter=4, leading=16)
H1       = _S("H1",      "Heading1",fontSize=14, textColor=AZUL, spaceBefore=14, spaceAfter=4, leading=18)
H2       = _S("H2",      "Heading2",fontSize=12, textColor=AZUL_MED, spaceBefore=10, spaceAfter=3, leading=15)
H3       = _S("H3",      "Heading3",fontSize=10, textColor=AZUL, spaceBefore=6, spaceAfter=2, leading=13)
BODY     = _S("BODY",    "Normal",  fontSize=9.5, leading=14, spaceAfter=4, alignment=TA_JUSTIFY)
BODYL    = _S("BODYL",   "Normal",  fontSize=9.5, leading=14, spaceAfter=4, alignment=TA_LEFT)
CODE     = _S("CODE",    "Normal",  fontSize=8, fontName="Courier",
              backColor=colors.HexColor("#F2F3F4"), leading=12,
              leftIndent=10, rightIndent=10, spaceBefore=4, spaceAfter=4)
CAPTION  = _S("CAPTION", "Normal",  fontSize=8, textColor=colors.grey,
              alignment=TA_CENTER, spaceAfter=6)
BULLET   = _S("BULLET",  "Normal",  fontSize=9.5, leading=14, leftIndent=14, spaceAfter=2)

# Estilos de célula de tabela — Paragraph dentro de Table renderiza HTML
CH      = _S("CH",  "Normal", fontSize=9,   fontName="Helvetica-Bold",
             textColor=BRANCO, alignment=TA_CENTER, leading=11, wordWrap="CJK")
CB      = _S("CB",  "Normal", fontSize=8.5, alignment=TA_CENTER, leading=11, wordWrap="CJK")
CBL     = _S("CBL", "Normal", fontSize=8.5, alignment=TA_LEFT,   leading=11, wordWrap="CJK")

# ── Helpers ──────────────────────────────────────────────────────────────────
def hr():
    return HRFlowable(width="100%", thickness=0.5, color=CINZA, spaceAfter=6, spaceBefore=6)

def sp(h=6):
    return Spacer(1, h)

def cor(texto, hexcor):
    return f'<font color="{hexcor}">{texto}</font>'

# ── Construtores de células ───────────────────────────────────────────────────
def ch(txt):
    """Célula de cabeçalho: Paragraph com estilo CH (branco, negrito, centralizado)."""
    return Paragraph(str(txt), CH)

def cb(txt):
    """Célula de corpo: Paragraph com estilo CB (preto, centralizado)."""
    return Paragraph(str(txt), CB)

def cbl(txt):
    """Célula de corpo alinhada à esquerda."""
    return Paragraph(str(txt), CBL)

# ── Tabela padrão ─────────────────────────────────────────────────────────────
_TS_BASE = TableStyle([
    ("BACKGROUND",     (0, 0), (-1,  0), AZUL),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [BRANCO, AZUL_PAL]),
    ("GRID",           (0, 0), (-1, -1), 0.4, CINZA),
    ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",     (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
    ("LEFTPADDING",    (0, 0), (-1, -1), 5),
    ("RIGHTPADDING",   (0, 0), (-1, -1), 5),
])

def tabela(cabecalho, linhas, col_widths=None, align_left_cols=None):
    """
    Monta uma Table com cabeçalho azul e linhas alternadas.
    cabecalho: lista de strings
    linhas: lista de listas de strings (podem conter tags HTML como <font color=...>)
    col_widths: lista de larguras em pontos; None = distribui igualmente
    align_left_cols: índices das colunas a alinhar à esquerda no corpo
    """
    if col_widths is None:
        w = PW / len(cabecalho)
        col_widths = [w] * len(cabecalho)

    align_left = set(align_left_cols or [])

    header_row = [ch(c) for c in cabecalho]
    body_rows = []
    for linha in linhas:
        row = []
        for i, cell in enumerate(linha):
            row.append(cbl(cell) if i in align_left else cb(cell))
        body_rows.append(row)

    data = [header_row] + body_rows
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(_TS_BASE)
    return t

# ══════════════════════════════════════════════════════════════════════════════
# CONTEÚDO
# ══════════════════════════════════════════════════════════════════════════════
story = []

# ── CAPA ──────────────────────────────────────────────────────────────────────
story += [
    sp(60),
    Paragraph("LACRIO IC", TITLE),
    Paragraph("Extração de Feições Supraglaciais com Deep Learning", SUBTITLE),
    sp(4), hr(), sp(4),
]

capa_data = [
    [Paragraph("Glaciar",         _S("KL","Normal",fontSize=9,fontName="Helvetica-Bold",textColor=AZUL)),
     Paragraph("Schiaparelli — Cordilheira Darwin, Terra do Fogo, Chile", _S("KV","Normal",fontSize=9))],
    [Paragraph("Bolsista",        _S("KL2","Normal",fontSize=9,fontName="Helvetica-Bold",textColor=AZUL)),
     Paragraph("Matheus Oliveira — Oceanologia, FURG", _S("KV2","Normal",fontSize=9))],
    [Paragraph("Orientador",      _S("KL3","Normal",fontSize=9,fontName="Helvetica-Bold",textColor=AZUL)),
     Paragraph("Prof. Dr. Jorge Arigony Neto", _S("KV3","Normal",fontSize=9))],
    [Paragraph("Laboratório",     _S("KL4","Normal",fontSize=9,fontName="Helvetica-Bold",textColor=AZUL)),
     Paragraph("LaCrio — Laboratório de Monitoramento da Criosfera", _S("KV4","Normal",fontSize=9))],
    [Paragraph("Financiamento",   _S("KL5","Normal",fontSize=9,fontName="Helvetica-Bold",textColor=AZUL)),
     Paragraph("CNPq — Bolsa IC (Set/2025 – Ago/2026)", _S("KV5","Normal",fontSize=9))],
    [Paragraph("Data do relatório",_S("KL6","Normal",fontSize=9,fontName="Helvetica-Bold",textColor=AZUL)),
     Paragraph(date.today().strftime("%d/%m/%Y"), _S("KV6","Normal",fontSize=9))],
]
t_capa = Table(capa_data, colWidths=[4.5*cm, 12.5*cm])
t_capa.setStyle(TableStyle([
    ("VALIGN",        (0,0),(-1,-1),"TOP"),
    ("TOPPADDING",    (0,0),(-1,-1), 5),
    ("BOTTOMPADDING", (0,0),(-1,-1), 5),
]))
story += [t_capa, PageBreak()]

# ── ÍNDICE ────────────────────────────────────────────────────────────────────
story.append(Paragraph("Índice", H1))
for item in [
    "1. Visão Geral do Projeto",
    "2. Área de Estudo e Dados",
    "3. Modelos Utilizados",
    "4. Scripts e Pipeline",
    "   4.1  config.py — Configurações centralizadas",
    "   4.2  01_create_tiles.py — Divisão em tiles",
    "   4.3  02_sam_interactive.py — Anotação interativa",
    "   4.4  03_finetune_sam.py — Fine-tuning SAM",
    "   4.5  03a_pretrain_satellite.py — Pré-treino em satélite",
    "   4.6  03_train_unet.py — Treinamento U-Net",
    "   4.7  04_inference.py — Inferência SAM",
    "   4.8  04_inference_unet.py — Inferência U-Net",
    "   4.9  05_reconstruct_mosaic.py — Reconstrução de mosaico",
    "   4.10 06_validate.py — Validação",
    "   4.11 annotate.py — Ferramenta de anotação manual",
    "   4.12 active_learning.py — Active Learning",
    "   4.13 shadow_utils.py — Detecção de sombra topográfica",
    "   4.14 Utilitários",
    "5. Resultados — Histórico e Estado Atual",
    "6. Comparação com Estado da Arte",
    "7. Próximos Passos",
]:
    story.append(Paragraph(item, BODYL))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# 1. VISÃO GERAL
# ══════════════════════════════════════════════════════════════════════════════
story += [Paragraph("1. Visão Geral do Projeto", H1), hr()]
story.append(Paragraph(
    "Este projeto de Iniciação Científica, financiado pelo CNPq e vinculado ao "
    "LaCrio/FURG, tem como objetivo mapear automaticamente <b>feições supraglaciais</b> "
    "— lagos, fendas (crevasses) e canais de degelo — em ortomosaicos de VANT "
    "(resolução 5,4 cm/pixel) do Glaciar Schiaparelli, na Cordilheira Darwin, Terra do Fogo, Chile.",
    BODY))
story.append(Paragraph(
    "O pipeline combina anotação assistida por SAM (Segment Anything Model) com "
    "treinamento supervisionado de uma <b>U-Net ResNet34</b> para segmentação semântica "
    "binária. Os resultados alimentarão análises de ablação superficial via DEMs "
    "multitemporais (2016–2022).",
    BODY))

story.append(Paragraph("Objetivos do edital", H2))
story.append(tabela(
    ["#", "Atividade", "Status"],
    [
        ["1", "Treinamento Agisoft Metashape", cor("Concluído", "#1E8449")],
        ["2", "Processamento dados VANT 2016–2022 (tiling RGB 512x512)", cor("Concluído", "#1E8449")],
        ["3", "Estimativa de ablação superficial (análise de DEMs)", cor("Não iniciado", "#922B21")],
        ["4", "Mapeamento hidrologia supraglacial (ML com SAM/U-Net)", cor("Em andamento", "#B7950B")],
    ],
    col_widths=[1*cm, 11.5*cm, 4.5*cm],
    align_left_cols=[1, 2],
))
story += [sp(10), PageBreak()]

# ══════════════════════════════════════════════════════════════════════════════
# 2. ÁREA DE ESTUDO E DADOS
# ══════════════════════════════════════════════════════════════════════════════
story += [Paragraph("2. Área de Estudo e Dados", H1), hr()]
story.append(Paragraph(
    "O <b>Glaciar Schiaparelli</b> (~54°S) é um dos maiores glaciares de montanha "
    "da Terra do Fogo chilena e encontra-se em acelerado recuo. Sua superfície "
    "apresenta lagos supraglaciais, fendas e canais de degelo que são "
    "indicadores sensíveis de mudanças climáticas.",
    BODY))

story.append(Paragraph("Dados disponíveis", H2))
story.append(tabela(
    ["Tipo", "Anos disponíveis", "Resolução", "Formato"],
    [
        ["Ortomosaicos RGB (VANT)", "2016, 2017, 2018, 2019, 2020", "5,4 cm/pixel", "GeoTIFF"],
        ["DEMs (VANT)", "2016, 2017, 2018, 2019, 2020, 2022", "22 cm/pixel", "GeoTIFF"],
    ],
    col_widths=[5*cm, 5.5*cm, 3.5*cm, 3*cm],
    align_left_cols=[0, 1],
))
story += [sp(6)]
story.append(Paragraph(
    "Os mosaicos são divididos em <b>tiles de 512x512 pixels</b> com sobreposição "
    "de 64 pixels. Tiles com menos de 70% de pixels válidos são descartados. "
    "Total gerado: <b>~21.798 tiles</b> (ano 2016).",
    BODY))

story.append(Paragraph("Feições-alvo", H2))
story.append(tabela(
    ["Feição", "Descrição", "Cor", "Área mín. (px)", "Área máx. (px)"],
    [
        ["Lakes",     "Lagos e poças supraglaciais",        "Azul",     "20",  "100.000"],
        ["Crevasses", "Fendas no gelo (fraturas mecânicas)", "Vermelho", "100", "10.000"],
        ["Channels",  "Canais de água de degelo",            "Ciano",    "200", "20.000"],
    ],
    col_widths=[2.5*cm, 6*cm, 2.5*cm, 3*cm, 3*cm],
    align_left_cols=[0, 1],
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# 3. MODELOS UTILIZADOS
# ══════════════════════════════════════════════════════════════════════════════
story += [Paragraph("3. Modelos Utilizados", H1), hr()]

# 3.1 SAM
story += [Paragraph("3.1  SAM — Segment Anything Model", H2)]
story.append(Paragraph(
    "<b>Origem:</b> Meta AI Research (Kirillov et al., 2023, ICCV). "
    "Repositório: github.com/facebookresearch/segment-anything.",
    BODY))
story.append(Paragraph(
    "SAM é um modelo de segmentação universal treinado em mais de 1 bilhão de máscaras. "
    "Usa um <b>Vision Transformer (ViT)</b> como encoder de imagem, um encoder de prompts "
    "(pontos, caixas, texto) e um decoder de máscaras leve.",
    BODY))

story.append(Paragraph("Variantes disponíveis:", H3))
story.append(tabela(
    ["Variante", "Parâmetros", "Checkpoint", "Velocidade", "Uso no projeto"],
    [
        ["ViT-B (base)", "~91M", "sam_vit_b_01ec64.pth", "Rápido", "Configuração padrão"],
        ["ViT-L (large)", "~308M", "sam_vit_l_0b3195.pth", "Médio", "Não usado"],
        ["ViT-H (huge)", "~636M", "sam_vit_h_4b8939.pth", "Lento", "Não usado"],
    ],
    col_widths=[2.5*cm, 2.5*cm, 5*cm, 2.5*cm, 4.5*cm],
    align_left_cols=[2, 4],
))
story += [sp(6)]
story.append(Paragraph(
    "<b>Como ajuda o projeto:</b> SAM foi usado em duas frentes — (1) ferramenta de "
    "anotação interativa no script 02_sam_interactive.py, onde o usuário clica em pontos "
    "e o modelo gera máscaras automáticas; e (2) modelo fine-tunable no 03_finetune_sam.py, "
    "onde o decoder é treinado com anotações manuais.",
    BODY))
story.append(Paragraph(
    "<b>Limitação descoberta:</b> SAM é um modelo interativo — no treino o prompt vem do "
    "ground truth (centróide exato da feição), mas na inferência qualquer estratégia "
    "automática de prompt é inferior. Isso causou um gap de Val Dice 0,87 no treino "
    "vs F1 0,12 na validação real — motivando a transição para U-Net.",
    BODY))

# 3.2 SAM-HQ
story += [sp(4), Paragraph("3.2  SAM-HQ — Segment Anything in High Quality", H2)]
story.append(Paragraph(
    "<b>Origem:</b> Ke et al. (2023), NeurIPS 2023. "
    "Repositório: github.com/SysCV/sam-hq. Checkpoint: sam_hq_vit_b.pth.",
    BODY))
story.append(Paragraph(
    "SAM-HQ introduz um <b>HQ-Output Token</b> adicional no decoder, produzindo máscaras "
    "com bordas mais nítidas — especialmente útil para crevasses (feições lineares de "
    "~1–5 px na resolução 5,4 cm/pixel). Drop-in replacement do SAM original.",
    BODY))
story.append(Paragraph(
    "<b>Resultado:</b> F1 lakes=0,265, crevasses=0,291, channels=0,451 "
    "(após fine-tuning, fev/2026). Val Dice alto no treino (0,87–0,90) não se "
    "traduziu em inferência automática eficaz.",
    BODY))

# 3.3 U-Net
story += [sp(4), Paragraph("3.3  U-Net com Encoder ResNet34", H2)]
story.append(Paragraph(
    "<b>Origem:</b> U-Net (Ronneberger et al., 2015) + encoder ResNet34 pré-treinado "
    "no ImageNet (He et al., 2016). Implementada em 03_train_unet.py.",
    BODY))
story.append(tabela(
    ["Componente", "Detalhes"],
    [
        ["Encoder", "ResNet34 pre-treinado no ImageNet — 4 blocos residuais (~21M params)"],
        ["Decoder", "4 blocos de upsampling com skip connections (ConvBlock duplo)"],
        ["Saida", "1 canal — sigmoid para probabilidade por pixel"],
        ["Entrada", "Tiles RGB 512x512 px normalizados com stats do ImageNet"],
        ["Params totais", "~24M (encoder 21M + decoder 3M)"],
        ["VRAM", "~2 GB com batch_size=4"],
        ["Inferencia", "~0,05 s/tile (60x mais rapido que SAM)"],
    ],
    col_widths=[4*cm, 13*cm],
    align_left_cols=[0, 1],
))
story += [sp(6)]
story.append(Paragraph(
    "<b>Vantagens sobre SAM:</b> nao precisa de prompt na inferencia (forward pass unico), "
    "treino e inferencia sao identicos (sem gap), e e 60x mais rapido.",
    BODY))
story.append(Paragraph(
    "<b>Tecnicas de treino:</b> Discriminative LR (encoder=1e-5, decoder=1e-4), "
    "Tversky Loss (BCE x 0,3 + Tversky x 0,7, fp_weight=0,7), "
    "Data Augmentation (Albumentations), Copy-Paste Augmentation, "
    "Early Stopping por val_f1 micro, backup automatico de checkpoints.",
    BODY))

# 3.4 LoRA
story += [sp(4), Paragraph("3.4  LoRA — Low-Rank Adaptation", H2)]
story.append(Paragraph(
    "<b>Origem:</b> Hu et al. (2021). Implementado nas projecoes QKV do ViT encoder "
    "do SAM. Rank=4, alpha=16, dropout=0,1.",
    BODY))
story.append(Paragraph(
    "<b>Resultado:</b> sem ganho mensuravel nos experimentos de fev/2026. "
    "Desativado por padrao (USE_LORA=False).",
    BODY))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# 4. SCRIPTS E PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
story += [Paragraph("4. Scripts e Pipeline", H1), hr()]
story.append(Paragraph(
    "O projeto segue um pipeline linear de 6 etapas numeradas, complementado "
    "por utilitarios de anotacao, active learning e analise. "
    "Todos os parametros globais sao centralizados em <b>config.py</b>.",
    BODY))

story.append(tabela(
    ["Etapa", "Script", "Entrada", "Saida"],
    [
        ["0 — Config",     "config.py",               "—",                    "Parametros globais"],
        ["1 — Tiling",     "01_create_tiles.py",       "Mosaico GeoTIFF",      "Tiles PNG 512x512"],
        ["2 — Anotacao",   "02_sam_interactive.py",    "Tiles PNG",            "Mascaras PNG (SAM)"],
        ["2b — Anotacao",  "annotate.py",              "Tiles + predicoes",    "Mascaras PNG (manual)"],
        ["3 — Treino SAM", "03_finetune_sam.py",        "Tiles + mascaras",     "Modelo SAM fine-tuned"],
        ["3a — Pre-treino","03a_pretrain_satellite.py", "Dataset satelite",     "Encoder pre-treinado"],
        ["3b — U-Net",     "03_train_unet.py",          "Tiles + mascaras",     "Modelo U-Net .pth"],
        ["4 — Inf. SAM",   "04_inference.py",           "Tiles + modelo SAM",   "Mascaras preditas"],
        ["4b — Inf. UNet", "04_inference_unet.py",      "Tiles + modelo U-Net", "Mascaras preditas"],
        ["5 — Mosaico",    "05_reconstruct_mosaic.py",  "Mascaras + indice",    "GeoTIFF final"],
        ["6 — Validacao",  "06_validate.py",            "Predicoes + GT",       "Metricas (F1, IoU)"],
        ["AL",             "active_learning.py",        "Modelo + tiles",       "CSV para anotacao"],
    ],
    col_widths=[3*cm, 5*cm, 4.5*cm, 4.5*cm],
    align_left_cols=[0, 1, 2, 3],
))
story.append(sp(10))

# ── 4.1 config.py ─────────────────────────────────────────────────────────────
story += [Paragraph("4.1  config.py — Configuracoes Centralizadas", H2)]
story.append(Paragraph(
    "Arquivo de configuracao unico que centraliza todos os parametros do projeto. "
    "Todos os outros scripts importam a classe <b>Config</b> deste modulo.",
    BODY))
story.append(tabela(
    ["Grupo", "Parametro", "Padrao", "Descricao"],
    [
        ["Modelo SAM", "MODEL_TYPE",               "vit_b",  "Variante do ViT encoder"],
        ["Modelo SAM", "USE_SAM_HQ",               "True",   "Usa SAM-HQ para bordas melhores"],
        ["Tiling",     "TILE_SIZE",                "512",    "Tamanho do tile em pixels"],
        ["Tiling",     "OVERLAP",                  "64",     "Sobreposicao entre tiles adjacentes"],
        ["Tiling",     "MIN_VALID_RATIO",           "0.7",   "Fracao minima de pixels validos"],
        ["Treino",     "LEARNING_RATE",             "1e-5",  "LR base do decoder"],
        ["Treino",     "EPOCHS",                   "50",    "Epocas maximas"],
        ["Treino",     "EARLY_STOPPING_PATIENCE",   "10",    "Paciencia do early stopping"],
        ["Treino",     "USE_AUGMENTATION",          "True",  "Augmentation Albumentations"],
        ["Treino",     "USE_COPY_PASTE",            "True",  "Copy-Paste augmentation"],
        ["LoRA",       "USE_LORA",                 "False", "LoRA no encoder (desativado)"],
        ["Inferencia", "USE_TTA",                  "True",  "Test-Time Augmentation (4 flips)"],
        ["Inferencia", "USE_MASK_REFINEMENT",       "True",  "Refinamento 2-pass"],
        ["Inferencia", "USE_SLOPE_FILTER",          "True",  "Filtro de slope via DEM"],
        ["Sombra",     "SHADOW_HILLSHADE_THRESHOLD","80",    "Limiar hillshade para sombra"],
    ],
    col_widths=[3*cm, 5.5*cm, 2.5*cm, 6*cm],
    align_left_cols=[0, 1, 3],
))
story.append(sp(10))

# ── 4.2 ───────────────────────────────────────────────────────────────────────
story += [Paragraph("4.2  01_create_tiles.py — Divisao em Tiles", H2)]
story.append(Paragraph(
    "Le os mosaicos GeoTIFF (varios GB) e os divide em tiles compativeis "
    "com SAM e U-Net. Cada tile e salvo como PNG com metadados de posicao "
    "em tiles_index.json para reconstrucao posterior.",
    BODY))
for s in [
    "Abre o mosaico com <b>rasterio</b> e le metadados (CRS, resolucao, dimensoes).",
    "Percorre posicoes (x, y) com passo = TILE_SIZE - OVERLAP.",
    "Verifica fracao de pixels validos (>= MIN_VALID_RATIO), normaliza para uint8.",
    "Salva o tile como PNG via OpenCV e registra posicao/geotransform no indice.",
]:
    story.append(Paragraph(f"• {s}", BULLET))
story.append(Paragraph(
    "Uso: python 01_create_tiles.py --year 2016   ou sem argumento para todos os anos. "
    "Flag --info mostra metadados sem processar.",
    CODE))
story.append(sp(8))

# ── 4.3 ───────────────────────────────────────────────────────────────────────
story += [Paragraph("4.3  02_sam_interactive.py — Anotacao Interativa com SAM", H2)]
story.append(Paragraph(
    "Interface grafica (OpenCV) para anotacao supervisionada de tiles. "
    "O usuario clica em pontos e o SAM gera mascaras automaticas que podem "
    "ser aceitas ou refinadas.",
    BODY))
for c in [
    "Clique esquerdo — prompt positivo (incluir regiao)",
    "Clique direito — prompt negativo (excluir regiao)",
    "Setas esquerda/direita — navegar entre tiles",
    "1/2/3 — selecionar feicao (lake / crevasse / channel)",
    "Enter — salvar mascara e avancar",
    "R — resetar prompts   |   M — modo multi-mask   |   Q — sair",
]:
    story.append(Paragraph(f"• {c}", BULLET))
story.append(sp(8))

# ── 4.4 ───────────────────────────────────────────────────────────────────────
story += [Paragraph("4.4  03_finetune_sam.py — Fine-tuning do SAM", H2)]
story.append(Paragraph(
    "Fine-tuning supervisionado do decoder do SAM com as anotacoes manuais. "
    "O encoder pode permanecer congelado ou ser adaptado via LoRA.",
    BODY))
for m in [
    "<b>Modo embeddings (padrao):</b> pre-computa embeddings do encoder em float16 "
    "e salva em disco. Treina APENAS o decoder. Consumo: ~1–2 GB VRAM.",
    "<b>Modo on-the-fly (--augment / --lora):</b> encoder roda durante o treino, "
    "permite augmentacoes e LoRA. Consumo: ~3–4 GB VRAM (batch_size=1).",
]:
    story.append(Paragraph(f"• {m}", BULLET))
story.append(sp(8))

# ── 4.5 ───────────────────────────────────────────────────────────────────────
story += [Paragraph("4.5  03a_pretrain_satellite.py — Pre-treino em Satelite", H2)]
story.append(Paragraph(
    "Script experimental para pre-treinar o encoder em datasets publicos de "
    "imagens de gelo de satelite (ex: SIGSPATIAL Ice Challenge) antes do "
    "fine-tuning nas imagens de drone, reduzindo o domain shift ImageNet->glaciar. "
    "Status: iniciado. Ganho potencial: +5–10 pp F1.",
    BODY))
story.append(sp(8))

# ── 4.6 ───────────────────────────────────────────────────────────────────────
story += [Paragraph("4.6  03_train_unet.py — Treinamento U-Net", H2)]
story.append(Paragraph(
    "Script principal de treinamento da U-Net ResNet34. Alternativa ao SAM "
    "para segmentacao semantica pura, sem necessidade de prompt.",
    BODY))
for f in [
    "<b>Discriminative LR:</b> encoder=1e-5, decoder=1e-4. Preserva features "
    "ImageNet enquanto adapta ao dominio glaciar.",
    "<b>Tversky Loss:</b> BCE x 0,3 + Tversky x 0,7 com fp_weight=0,7. "
    "Reduziu oversegmentacao de 8 para 0 tiles.",
    "<b>Data Augmentation:</b> flip, rotacao, brilho/contraste via Albumentations. "
    "Copy-Paste multiplica o dataset efetivo por 2–3x.",
    "<b>Criterio de checkpoint:</b> val_f1 micro (nao val_dice, que e inflacionado). "
    "Backup automatico com timestamp antes de sobrescrever.",
]:
    story.append(Paragraph(f"• {f}", BULLET))
story.append(sp(8))

# ── 4.7 ───────────────────────────────────────────────────────────────────────
story += [Paragraph("4.7  04_inference.py — Inferencia SAM Fine-tuned", H2)]
story.append(Paragraph(
    "Aplica o SAM fine-tuned em todos os ~22.000 tiles com grade densa de "
    "pontos de prompt. Inclui pos-processamento avancado:",
    BODY))
for i in [
    "<b>TTA:</b> media de 4 predicoes (original, hflip, vflip, rot180). +2–4 pp IoU.",
    "<b>Refinamento 2-pass:</b> mascara do 1o pass vira prompt para o 2o.",
    "<b>Filtro de slope via DEM:</b> rejeita lagos em areas com inclinacao >15 graus.",
    "<b>Filtros espectrais:</b> blue_ratio, dark_threshold para lagos; "
    "min_aspect para crevasses/canais.",
]:
    story.append(Paragraph(f"• {i}", BULLET))
story.append(sp(8))

# ── 4.8 ───────────────────────────────────────────────────────────────────────
story += [Paragraph("4.8  04_inference_unet.py — Inferencia U-Net", H2)]
story.append(Paragraph(
    "Forward pass unico por tile — ~0,05 s/tile (60x mais rapido que SAM). "
    "Reutiliza filtros de pos-processamento do 04_inference.py. "
    "Integra mascara de sombra (shadow_utils.py) e suporta --validate "
    "para calculo de metricas sobre tiles anotados.",
    BODY))
story.append(sp(8))

# ── 4.9 ───────────────────────────────────────────────────────────────────────
story += [Paragraph("4.9  05_reconstruct_mosaic.py — Reconstrucao de Mosaico", H2)]
story.append(Paragraph(
    "Combina mascaras individuais de tiles em um mosaico GeoTIFF "
    "georreferenciado, pronto para analise no QGIS ou ArcGIS.",
    BODY))
for s in [
    "Carrega tiles_index.json para obter posicoes originais de cada tile.",
    "Cria array vazio com as dimensoes do mosaico original.",
    "Insere cada mascara na posicao correta; sobreposicoes usam operador maximo.",
    "Salva como GeoTIFF com CRS e geotransform originais.",
]:
    story.append(Paragraph(f"• {s}", BULLET))
story.append(sp(8))

# ── 4.10 ──────────────────────────────────────────────────────────────────────
story += [Paragraph("4.10  06_validate.py — Validacao e Metricas", H2)]
story.append(Paragraph(
    "Calcula metricas de segmentacao tile a tile e gera relatorio JSON + grafico PNG. "
    "Metricas: Precision, Recall, F1, IoU, Dice — reportados em micro "
    "(soma de TP/FP/FN) e macro (media por tile).",
    BODY))
story.append(sp(8))

# ── 4.11 ──────────────────────────────────────────────────────────────────────
story += [Paragraph("4.11  annotate.py — Ferramenta de Anotacao Manual", H2)]
story.append(Paragraph(
    "Interface Tkinter avancada para anotacao e revisao de mascaras. "
    "Exibe o tile com a predicao do modelo como sugestao e permite "
    "edicao pixel a pixel com pincel.",
    BODY))
for a in [
    "Pincel com tamanho ajustavel via scroll; borracha com clique direito.",
    "Zoom 1x–64x (Ctrl+Scroll) com pan (botao do meio + arrastar).",
    "Tecla A: aceitar predicao como anotacao.",
    "Tecla R: rejeitar tile (salvar mascara vazia = sem feicao).",
    "Tecla Z: desfazer ultima pincelada.",
    "Modo --review: revisao de tiles ja anotados para corrigir erros.",
    "Carrega CSV do active_learning.py para priorizar tiles mais incertos.",
]:
    story.append(Paragraph(f"• {a}", BULLET))
story.append(sp(8))

# ── 4.12 ──────────────────────────────────────────────────────────────────────
story += [Paragraph("4.12  active_learning.py — Active Learning", H2)]
story.append(Paragraph(
    "Seleciona os tiles mais informativos para anotacao manual, "
    "maximizando o ganho por hora de anotacao.",
    BODY))
for s in [
    "<b>uncertainty (padrao para lakes):</b> tiles onde o modelo detectou algo "
    "mas com baixa confianca. Score = pred_area x (1 - max_prob).",
    "<b>random:</b> amostragem aleatoria dentro do glaciar, filtrando vegetacao "
    "(G-B > 15) e NoData. Preferida para crevasses em anos sem dados de treino.",
]:
    story.append(Paragraph(f"• {s}", BULLET))
story.append(Paragraph(
    "<b>Licao aprendida:</b> uncertainty sampling para crevasses selecionou "
    "tiles ambiguos demais — modelo aprendeu a ser ultra-conservador "
    "(Recall < 0,20 em 2017/2019). Random sampling e mais seguro quando o "
    "modelo ainda nao tem sinal claro na feicao alvo.",
    BODY))
story.append(sp(8))

# ── 4.13 ──────────────────────────────────────────────────────────────────────
story += [Paragraph("4.13  shadow_utils.py — Deteccao de Sombra Topografica", H2)]
story.append(Paragraph(
    "Gera mascaras de sombra a partir do DEM para remover falsos positivos "
    "na deteccao de lagos (sombras escuras sao confundidas com agua).",
    BODY))
for s in [
    "<b>Hillshade (Horn, 1981):</b> mesmo algoritmo do GDAL gdaldem hillshade.",
    "<b>Multiplos angulos solares:</b> azimutes [330, 0, 30 graus] e altitudes "
    "[30, 40, 50 graus] tipicos para ~54 graus S no verao austral.",
    "<b>Intersecao conservadora:</b> pixel = sombra somente se em sombra em "
    "TODOS os angulos — minimiza falsos alarmes.",
    "<b>Filtro de textura:</b> sombras sao uniformes (baixa variancia); "
    "lagos tem reflexos — variancia distingue os dois.",
]:
    story.append(Paragraph(f"• {s}", BULLET))
story.append(sp(8))

# ── 4.14 Utilitários ──────────────────────────────────────────────────────────
story += [Paragraph("4.14  Utilitarios e Scripts de Suporte", H2)]
story.append(tabela(
    ["Script", "Funcao"],
    [
        ["check_empty_masks.py",
         "Detecta e remove mascaras com zero pixels positivos, evitando que "
         "tiles zerados sejam contados como positivos no treino."],
        ["convert_qgis_to_masks.py",
         "Converte shapefiles de anotacao do QGIS para mascaras PNG "
         "no formato esperado pelo pipeline."],
        ["debug_tiles.py",
         "Ferramenta de diagnostico visual para inspecao de tiles, "
         "sobreposicao de mascaras e analise de erros."],
        ["test_strategies.py",
         "Testa configuracoes de prompt e pos-processamento para "
         "otimizacao da inferencia SAM."],
        ["prepare_sigspatial.py",
         "Prepara o dataset SIGSPATIAL Ice Challenge para pre-treino "
         "do encoder em imagens de gelo de satelite."],
        ["run_ablation_stage34.sh",
         "Script shell para execucao automatica de ablacoes comparando "
         "configuracoes de treino (stages 3 e 4)."],
    ],
    col_widths=[5*cm, 12*cm],
    align_left_cols=[0, 1],
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# 5. RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════
story += [Paragraph("5. Resultados — Historico e Estado Atual", H1), hr()]

story += [Paragraph("5.1  Metas do Projeto (edital CNPq)", H2)]
story.append(tabela(
    ["Feicao", "F1 alvo", "IoU alvo"],
    [
        ["Lakes",     "85–90%", "75–85%"],
        ["Crevasses", "80–85%", "70–80%"],
        ["Channels",  "75–85%", "65–75%"],
    ],
    col_widths=[5*cm, 6*cm, 6*cm],
))
story.append(sp(10))

story += [Paragraph("5.2  Evolucao Cronologica dos Resultados", H2)]

# Primeiros resultados
story.append(Paragraph(
    cor("Primeiros resultados SAM (fev/2026)", "#922B21"), H3))
story.append(Paragraph(
    "Baseline Dice+BCE sem augmentation: F1 lakes=0,00 (deteccoes em locais errados). "
    "v1 (normalizacao SAM + grade 8x8): F1=0,00. "
    "v2 (+amostras negativas): F1=0,19, FPs por sombra. "
    "v3 (+mascara de sombra DEM): melhora incremental.",
    BODY))

# Ablacao SAM-HQ
story.append(Paragraph(cor("Ablacao SAM-HQ (04/fev/2026)", "#B7950B"), H3))
story.append(tabela(
    ["Feicao", "HQ Only (F1/IoU)", "HQ + Aug (F1/IoU)", "HQ + Aug + LoRA (F1/IoU)"],
    [
        ["Lakes",     "0,179 / 0,098", "0,286 / 0,167", "0,286 / 0,167"],
        ["Crevasses", "0,255 / 0,146", "0,276 / 0,160", "0,276 / 0,160"],
        ["Channels",  "0,365 / 0,223", "0,533 / 0,364", "0,533 / 0,364"],
    ],
    col_widths=[3*cm, 4.5*cm, 4.5*cm, 5*cm],
))
story.append(Paragraph(
    "Augmentation: +60% F1 em lakes. LoRA: sem efeito mensuravel.",
    CAPTION))
story.append(sp(8))

# SAM-HQ Tuning CLI
story.append(Paragraph(cor("SAM-HQ Tuning com CLI (08/fev/2026)", "#B7950B"), H3))
story.append(tabela(
    ["Feicao", "Configuracao", "P", "R", "F1", "IoU"],
    [
        ["Lakes",     "max, iou=0,6, thr=0,60", "0,181", "0,493", "0,265", "0,153"],
        ["Crevasses", "max, iou=0,70, thr=0,60", "0,221", "0,428", "0,291", "0,170"],
        ["Channels",  "max, iou=0,60, thr=0,50", "0,373", "0,571", "0,451", "0,291"],
    ],
    col_widths=[2.5*cm, 6*cm, 2*cm, 2*cm, 2*cm, 2.5*cm],
    align_left_cols=[0, 1],
))
story.append(sp(8))

# Gap treino-inferencia
story.append(Paragraph(cor("Diagnostico: Gap Treino-Inferencia do SAM (07/abr/2026)", "#922B21"), H3))
story.append(Paragraph(
    "SAM atingiu Val Dice 0,87 no treino, mas F1 0,12 na validacao real. "
    "Causa: no treino o prompt vem do GT (centroide exato); na inferencia "
    "automatica qualquer estrategia e inferior.",
    BODY))
story.append(tabela(
    ["Categoria dos tiles avaliados", "Qtd", "%"],
    [
        ["Bons (overlap razoavel)", "20", "39%"],
        ["Zero overlap (pred fora do GT)", "13", "25%"],
        ["Oversegmentacao (>5x GT)", "13", "25%"],
        ["Sem predicao alguma", "5", "10%"],
    ],
    col_widths=[9*cm, 2.5*cm, 5.5*cm],
    align_left_cols=[0],
))
story.append(Paragraph(
    "GT total: 48.687 px | Pred total: 247.395 px (5,1x mais area que real). "
    "Decisao: migrar para U-Net.",
    CAPTION))
story.append(sp(8))

# U-Net Lakes Exp C
story.append(Paragraph(cor("U-Net Lakes — Exp. C (discriminative LR) (15/abr/2026)", "#1E8449"), H3))
story.append(tabela(
    ["Ano", "Tiles", "F1", "P", "R", "IoU", "Overseg"],
    [
        ["2016", "34", cor("0,690", "#1E8449"), "0,635", "0,756", "0,527", "0"],
        ["2017", "30", cor("0,518", "#B7950B"), "0,442", "0,627", "0,350", "1"],
        ["2018", "30", cor("0,719", "#1E8449"), "0,635", "0,828", "0,561", "0"],
        ["2019", "30", cor("0,570", "#B7950B"), "0,461", "0,746", "0,398", "1"],
    ],
    col_widths=[2*cm, 2*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 3*cm],
))
story.append(Paragraph(
    "Melhor resultado em lakes. Ganho vs Exp. A: +29,9 pp (2018), +39,4 pp (2019).",
    CAPTION))
story.append(sp(8))

# U-Net Crevasses revisao
story.append(Paragraph(cor("U-Net Crevasses — Revisao completa 4 anos (20/abr/2026)", "#1E8449"), H3))
story.append(Paragraph(
    "189 tiles positivos + 150 hard negatives. Tversky fp_weight=0,7. "
    "200 epocas, melhor epoca 55. "
    "Dataset: 2016(31) · 2017(44+36HN) · 2018(71+78HN) · 2019(43+36HN).",
    BODY))
story.append(tabela(
    ["Ano", "Tiles", "F1", "P", "R", "IoU", "Overseg", "Zero overlap"],
    [
        ["2016", "31",  cor("0,630", "#1E8449"), "0,756", "0,541", "0,460", "0", "2"],
        ["2017", "80",  cor("0,473", "#B7950B"), "0,692", "0,360", "0,310", "0", "12"],
        ["2018", "149", cor("0,511", "#B7950B"), "0,600", "0,445", "0,343", "0", "1"],
        ["2019", "79",  cor("0,531", "#B7950B"), "0,807", "0,396", "0,361", "0", "2"],
    ],
    col_widths=[1.5*cm, 1.5*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2.5*cm, 3.5*cm],
))
story.append(Paragraph(
    "Overseg=0 em todos os anos (vs 8 no baseline). "
    "Recall baixo em 2017/2019: efeito fp_weight=0,7.",
    CAPTION))
story.append(sp(8))

# AL Uncertainty descartado
story.append(Paragraph(cor("AL Uncertainty — Descartado (22/abr/2026)", "#922B21"), H3))
story.append(Paragraph(
    "Active learning com uncertainty sampling (+40 tiles 2017/2019, total 419 pos). "
    "Re-treino completo 200 epocas (patience=100), best ep.29.",
    BODY))
story.append(tabela(
    ["Ano", "F1", "P", "R", "Diagnostico"],
    [
        ["2016", "0,577", "0,664", "0,510", "Regressao vs referencia"],
        ["2017", cor("0,257", "#922B21"), "0,772", cor("0,154", "#922B21"), "Recall colapsado"],
        ["2018", "0,511", "0,649", "0,421", "Neutro"],
        ["2019", cor("0,317", "#922B21"), "0,816", cor("0,196", "#922B21"), "Recall colapsado"],
    ],
    col_widths=[2*cm, 2.5*cm, 2.5*cm, 2.5*cm, 7.5*cm],
    align_left_cols=[4],
))
story.append(Paragraph(
    "Decisao: descartar. Restaurar ep.63 como producao. "
    "Uncertainty sampling descartado definitivamente para crevasses.",
    CAPTION))
story.append(sp(8))

# Estado atual
story += [Paragraph("5.3  Estado Atual — Modelos de Producao", H2)]
story.append(tabela(
    ["Feicao", "Modelo", "Ano", "F1", "IoU", "Checkpoint"],
    [
        ["Lakes",     "U-Net ResNet34", "2016", "0,690", "0,527", "ep.59 (Exp. C)"],
        ["Lakes",     "U-Net ResNet34", "2018", "0,719", "0,561", "ep.59 (Exp. C)"],
        ["Crevasses", "U-Net ResNet34", "2016", "0,630", "0,460", "ep.63"],
        ["Crevasses", "U-Net ResNet34", "2017", "0,476", "—",     "ep.63"],
        ["Crevasses", "U-Net ResNet34", "2018", "0,513", "—",     "ep.63"],
        ["Crevasses", "U-Net ResNet34", "2019", "0,532", "—",     "ep.63"],
    ],
    col_widths=[2.5*cm, 3.5*cm, 1.5*cm, 2*cm, 2*cm, 5.5*cm],
    align_left_cols=[0, 1, 5],
))
story.append(sp(10))

story += [Paragraph("5.4  Cobertura Reconstruida (Lagos, SAM)", H2)]
story.append(tabela(
    ["Ano", "Tiles usados", "Pixels de lago", "Cobertura (%)"],
    [
        ["2017", "84",  "794.208",   "0,12%"],
        ["2018", "46",  "663.568",   "0,11%"],
        ["2019", "281", "3.496.720", "0,24%"],
        ["2020", "102", "1.153.584", "0,05%"],
    ],
    col_widths=[3*cm, 4*cm, 5.5*cm, 4.5*cm],
))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# 6. ESTADO DA ARTE
# ══════════════════════════════════════════════════════════════════════════════
story += [Paragraph("6. Comparacao com Estado da Arte", H1), hr()]
story.append(tabela(
    ["Metodo", "Feicao", "Resolucao", "F1", "IoU"],
    [
        ["Chai et al. (2025) — SAM fine-tuned", "Lagos", "10 m (Sentinel-2)", "87,8%", "—"],
        ["Wallace et al. (2025) — SAM 2 zero-shot", "Crevasses", "~5–10 cm (drone)", "—", "28%"],
        ["Nosso — SAM-HQ + Aug", "Lakes", "5,4 cm (drone)", "28,6%", "16,7%"],
        ["Nosso — SAM-HQ + Aug", "Crevasses", "5,4 cm (drone)", "27,6%", "16,0%"],
        ["Nosso — SAM-HQ + Aug", "Channels", "5,4 cm (drone)", "53,3%", "36,4%"],
        ["Nosso — U-Net Exp. C", "Lakes", "5,4 cm (drone)", "69% (2016) / 72% (2018)", "0,53 / 0,56"],
        ["Nosso — U-Net ep.63", "Crevasses", "5,4 cm (drone)", "63% (2016)", "0,46"],
    ],
    col_widths=[5.5*cm, 2.5*cm, 3*cm, 3.5*cm, 2.5*cm],
    align_left_cols=[0],
))
story.append(sp(8))
story.append(Paragraph(
    "Observacoes: (1) Chai et al. usam Sentinel-2 (10 m) com abundancia de dados "
    "publicos anotados — contexto diferente do nosso (drone 5,4 cm, ~200 tiles proprios). "
    "(2) Wallace et al. alcancam IoU=0,28 zero-shot com SAM 2, indicando que nosso F1 "
    "de crevasses (IoU=0,16) ainda esta abaixo do potencial — ha margem com mais dados.",
    BODY))
story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════════════════
# 7. PROXIMOS PASSOS
# ══════════════════════════════════════════════════════════════════════════════
story += [Paragraph("7. Proximos Passos", H1), hr()]

story += [Paragraph("7.1  Crevasses (prioridade imediata)", H2)]
for p in [
    "Restaurar ep.63 como modelo de producao "
    "(cp unet_crevasses_best_20260421_2204.pth unet_crevasses_best.pth).",
    "Active learning --strategy random em 2017 e 2019: ~40 tiles com amostragem "
    "aleatoria para dataset mais diverso (nao uncertainty).",
    "Hard FNs 2016: anotar tiles similares a tile_000182, tile_000635, "
    "tile_000887, tile_000342 (crevasses sutis que o modelo nao ativa).",
]:
    story.append(Paragraph(f"• {p}", BULLET))

story += [sp(4), Paragraph("7.2  Lakes (melhorias pendentes)", H2)]
for p in [
    "Hard negatives explicitos: mascaras GT=0 para tile_000929, tile_001056, "
    "tile_001057, tile_001224, tile_001286 (FPs estruturais em 2018).",
    "Active learning 2017–2019: expandir dataset alem das 34 amostras atuais.",
    "Filtro de slope via DEM: rejeitar lagos em areas com slope >15 graus.",
]:
    story.append(Paragraph(f"• {p}", BULLET))

story += [sp(4), Paragraph("7.3  Analise de DEMs / Ablacao (edital — nao iniciado)", H2)]
story.append(Paragraph(
    "Atividades 3 e 4 do edital CNPq — analise de DEMs multitemporais (2016–2022) "
    "para estimar perda de volume, taxa de ablacao e evolucao da hidrologia supraglacial.",
    BODY))
for p in [
    "dH (diferenca de elevacao) entre pares de anos: 2016, 2017, 2018, 2019, 2020, 2022.",
    "Calculo de volume perdido por segmento do glaciar.",
    "Correlacao entre area de lakes/canais e taxa de ablacao.",
    "Relatorio final de IC (prazo: agosto/2026).",
]:
    story.append(Paragraph(f"• {p}", BULLET))

story += [sp(4), Paragraph("7.4  Melhorias de modelo (medio prazo)", H2)]
for p in [
    "Pre-treino em datasets publicos (SIGSPATIAL): reduzir domain shift ImageNet/glaciar.",
    "Boundary Loss: para crevasses (feicoes lineares finas).",
    "CRF pos-processamento: refinar bordas via Dense CRF.",
    "SAM 2 Hiera-Tiny: potencial zero-shot superior ao SAM 1 (ver Wallace et al., 2025).",
]:
    story.append(Paragraph(f"• {p}", BULLET))

story += [
    sp(20), hr(),
    Paragraph(
        f"Relatorio gerado em {date.today().strftime('%d/%m/%Y')} — "
        "LACRIO IC / LaCrio-FURG / CNPq",
        CAPTION),
]

# ══════════════════════════════════════════════════════════════════════════════
# BUILD
# ══════════════════════════════════════════════════════════════════════════════
doc = SimpleDocTemplate(
    OUTPUT_PATH,
    pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
    title="Relatorio LACRIO IC — Extracao de Feicoes Supraglaciais",
    author="Matheus Oliveira / LaCrio-FURG",
    subject="Segmentacao de feicoes supraglaciais com SAM e U-Net",
)
doc.build(story)
print(f"PDF gerado: {OUTPUT_PATH}")
