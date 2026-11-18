#!/usr/bin/env bash
# Gera Relatorio_Final_IC_LACRIO_2026.docx/.pdf a partir de relatorio.md.
# Mapas em páginas A4 horizontais (quebras de seção em openxml no .md).
set -euo pipefail
cd "$(dirname "$0")"
PANDOC=${PANDOC:-$HOME/miniconda3/envs/sam_glaciar/bin/pandoc}
"$PANDOC" relatorio.md -o Relatorio_Final_IC_LACRIO_2026.docx --resource-path=. --reference-doc=ref_a4.docx
python3 ajusta_mapas_paisagem.py Relatorio_Final_IC_LACRIO_2026.docx
soffice --headless --convert-to pdf Relatorio_Final_IC_LACRIO_2026.docx >/dev/null
echo "ok: $(pdfinfo Relatorio_Final_IC_LACRIO_2026.pdf | grep Pages)"
