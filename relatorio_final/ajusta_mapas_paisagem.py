"""Aumenta os mapas das páginas horizontais no .docx gerado pelo pandoc.

O pandoc limita toda imagem à largura do texto da página em pé (16 cm), então
os mapas das seções em paisagem saem pequenos. Este script localiza cada mapa
pelo SHA-1 do arquivo de mídia embutido (igual ao do PNG original)
e reescreve o tamanho no desenho (wp:extent e a:ext).
"""
import hashlib, re, shutil, sys, tempfile, zipfile
from pathlib import Path
from PIL import Image

FIG = Path(__file__).resolve().parent.parent / "results/figures/relatorio"
EMU_CM = 360000
# mapa -> tamanho alvo (cm) na página A4 horizontal (área útil 26,7 × 18 cm)
MAPAS = {
    "mapa_area_estudo_qgis.png": {"altura": 15.0},
    "mapa_folds_cv.png": {"altura": 14.2},
    "mapa_fendas_2019_qgis.png": {"altura": 15.0},
    "mapa_mascaras_4anos_qgis.png": {"altura": 15.0},
    "mapa_densidade.png": {"altura": 15.0},
    "mapa_dh.png": {"largura": 26.0},
}


def tamanho_emu(png, alvo):
    w, h = Image.open(png).size
    if "altura" in alvo:
        cy = alvo["altura"] * EMU_CM
        cx = cy * w / h
    else:
        cx = alvo["largura"] * EMU_CM
        cy = cx * h / w
    return int(cx), int(cy)


def main(docx):
    docx = Path(docx)
    sha = {hashlib.sha1((FIG / n).read_bytes()).hexdigest(): n for n in MAPAS}
    with zipfile.ZipFile(docx) as z:
        itens = {i.filename: z.read(i.filename) for i in z.infolist()}
    rels = itens["word/_rels/document.xml.rels"].decode()
    rid_mapa = {}
    for rid, alvo in re.findall(r'Id="(rId\d+)"[^>]*Target="(media/[^"]+)"', rels):
        h = hashlib.sha1(itens["word/" + alvo]).hexdigest()
        if h in sha:
            rid_mapa[rid] = sha[h]
    doc = itens["word/document.xml"].decode()
    n = 0

    def troca(m):
        nonlocal n
        bloco = m.group(0)
        r = re.search(r'r:embed="(rId\d+)"', bloco)
        if not r or r.group(1) not in rid_mapa:
            return bloco
        nome = rid_mapa[r.group(1)]
        cx, cy = tamanho_emu(FIG / nome, MAPAS[nome])
        bloco = re.sub(r'<wp:extent cx="\d+" cy="\d+"', f'<wp:extent cx="{cx}" cy="{cy}"', bloco)
        bloco = re.sub(r'<a:ext cx="\d+" cy="\d+"', f'<a:ext cx="{cx}" cy="{cy}"', bloco)
        n += 1
        return bloco

    doc = re.sub(r"<w:drawing>.*?</w:drawing>", troca, doc, flags=re.S)
    itens["word/document.xml"] = doc.encode()
    tmp = Path(tempfile.mkstemp(suffix=".docx")[1])
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for nome, dados in itens.items():
            z.writestr(nome, dados)
    shutil.move(tmp, docx)
    print(f"{n} mapas redimensionados de {len(MAPAS)} esperados")
    if n != len(MAPAS):
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Relatorio_Final_IC_LACRIO_2026.docx")
