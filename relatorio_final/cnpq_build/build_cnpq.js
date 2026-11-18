// Relatorio Final PIBIC/PIBITI CNPq - FURG (modelo_relatorio_final_cnpq_ic_it.doc), gerado com docx-js.
// Layout replicado do modelo: A4, margens 2 cm (inferior 3 cm), Arial 10, espaco simples, secoes em caixas.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, WidthType, BorderStyle, AlignmentType,
  ImageRun, PageBreak, Footer, PageNumber, VerticalAlign, ShadingType,
} = require("docx");

const C = require("./conteudo_cnpq.js");
const MEDIA = path.join(__dirname, "media");
const OUT = process.argv[2] || path.join(__dirname, "..", "Relatorio_Final_CNPq_FURG_2026.docx");

const FONT = "Arial";
const SZ = 20;        // 10 pt
const SZ_SMALL = 16;  // 8 pt
const SZ_TITLE = 24;  // 12 pt
const W = 9638;       // largura util (11906 - 2*1134)
const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const BOX = { top: border, bottom: border, left: border, right: border };
const NOBOX = { top: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  left: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" }, right: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" } };

// ---- runs com marcacao simples: **negrito**, *italico*, ^{sobrescrito}
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, size: SZ, ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), font: FONT, size: SZ, bold: true, ...base }));
    else if (tok.startsWith("^{")) out.push(new TextRun({ text: tok.slice(2, -1), font: FONT, size: SZ, superScript: true, ...base }));
    else out.push(new TextRun({ text: tok.slice(1, -1), font: FONT, size: SZ, italics: true, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, size: SZ, ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text, opts.run || {}), alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { before: opts.before ?? 0, after: opts.after ?? 40, line: 240 }, indent: opts.indent });
const PB = (text, opts = {}) => new Paragraph({ children: [new TextRun({ text, font: FONT, size: opts.size || SZ, bold: true })],
  alignment: opts.align || AlignmentType.LEFT, spacing: { before: opts.before ?? 0, after: opts.after ?? 60, line: 240 } });
const EMPTY = () => new Paragraph({ children: [new TextRun({ text: "", font: FONT, size: 12 })], spacing: { after: 0, line: 120 } });

function cell(children, width, opts = {}) {
  return new TableCell({ children, width: { size: width, type: WidthType.DXA }, borders: opts.borders || BOX,
    margins: { top: 20, bottom: 20, left: 80, right: 80 }, verticalAlign: opts.valign || VerticalAlign.TOP,
    columnSpan: opts.span, shading: opts.shade ? { type: ShadingType.CLEAR, fill: opts.shade, color: "auto" } : undefined });
}

// ---- cabecalho institucional (tabela sem bordas: logo | texto | logo)
function headerTable() {
  const img1 = fs.readFileSync(path.join(MEDIA, "image1.png"));
  const img2 = fs.readFileSync(path.join(MEDIA, "image2.png"));
  const lines = ["SERVIÇO PÚBLICO FEDERAL", "MINISTÉRIO DA EDUCAÇÃO", "UNIVERSIDADE FEDERAL DO RIO GRANDE - FURG",
    "PRÓ-REITORIA DE PESQUISA E PÓS-GRADUAÇÃO - PROPESP", "DIRETORIA DE PESQUISA - DIPESQ"];
  const mid = lines.map(t => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0, line: 240 },
    children: [new TextRun({ text: t, font: FONT, size: SZ_SMALL, bold: true })] }));
  mid.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0, line: 240 },
    children: [new TextRun({ text: "Av. Itália, km 8 - Bairro Carreiros, Rio Grande – RS, CEP: 96.203-900", font: FONT, size: SZ_SMALL })] }));
  mid.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0, line: 240 },
    children: [new TextRun({ text: "Fone: (53) 3233-6821 FAX: 3233-6822 — E-mail: dipesq@furg.br - Homepage: http://www.propesp.furg.br", font: FONT, size: SZ_SMALL })] }));
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [2313, 5186, 2139], borders: NOBOX,
    rows: [new TableRow({ children: [
      cell([new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "png", data: img1, transformation: { width: 115, height: 49 } })] })], 2313, { borders: NOBOX, valign: VerticalAlign.CENTER }),
      cell(mid, 5186, { borders: NOBOX, valign: VerticalAlign.CENTER }),
      cell([new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "png", data: img2, transformation: { width: 56, height: 59 } })] })], 2139, { borders: NOBOX, valign: VerticalAlign.CENTER }),
    ] })] });
}

function folhaTitle(n, limite) {
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 160, after: 60 },
      children: [new TextRun({ text: `RELATÓRIO FINAL PIBIC/PIBITI CNPq – FOLHA ${n}`, font: FONT, size: SZ_TITLE, bold: true })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
      children: [new TextRun({ text: limite, font: FONT, size: SZ, bold: true, italics: true })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
      children: [new TextRun({ text: "PREENCHIMENTO EM FOLHA A4, FONTE ARIAL 10, ESPAÇO SIMPLES.", font: FONT, size: SZ_SMALL })] }),
  ];
}

// ---- tabela de identificacao (rotulo em negrito + valor)
function idTable(withArea) {
  const L = (t) => PB(t);
  const V = (t) => P(t, { align: AlignmentType.LEFT, after: 0 });
  const row2 = (l1, v1, l2, v2) => [
    new TableRow({ children: [cell([L(l1)], 7138), cell([L(l2)], 2500)] }),
    new TableRow({ children: [cell([V(v1)], 7138), cell([V(v2)], 2500)] }),
  ];
  const row1 = (l, v) => [
    new TableRow({ children: [cell([L(l)], W, { span: 2 })] }),
    new TableRow({ children: [cell([V(v)], W, { span: 2 })] }),
  ];
  const rows = [
    ...row2("Orientador(a):", C.id.orientador, "Sigla Dept./Unid.:", C.id.sigla),
    ...row2("Nome do Aluno(a) Bolsista:", C.id.aluno, "N° do Processo:", C.id.processo),
    ...row2("Matrícula/Curso:", C.id.matricula, "Vigência da Bolsa:", C.id.vigencia),
    ...row1("Título do Projeto:", C.id.titulo),
  ];
  if (withArea) rows.push(...row1("Área/Subárea do CNPq:", C.id.area));
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [7138, 2500], rows });
}

// ---- caixa de secao: titulo em negrito + blocos (paragrafo | tabela | imagem)
function block(b) {
  if (typeof b === "string") return [P(b)];
  if (b.bullets) return b.bullets.map(t => new Paragraph({ children: [new TextRun({ text: "– ", font: FONT, size: SZ }), ...runs(t)],
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 40, line: 240 }, indent: { left: 200, hanging: 200 } }));
  if (b.table) {
    const cw = b.widths; const T = b.table;
    const mk = (r, head, last) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell([new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      spacing: { after: 0, line: 240 }, children: runs(t, { size: SZ_SMALL, bold: head }) })], cw[i], { shade: head ? "E7E6E6" : undefined })) });
    return [
      new Paragraph({ children: runs(b.caption), alignment: AlignmentType.LEFT, spacing: { before: 60, after: 40, line: 240 } }),
      new Table({ width: { size: cw.reduce((a, c) => a + c, 0), type: WidthType.DXA }, columnWidths: cw, rows: [mk(T[0], true, false), ...T.slice(1).map((r, k) => mk(r, false, k === T.length - 2))] }),
      new Paragraph({ children: [new TextRun({ text: b.note || "", font: FONT, size: SZ_SMALL, italics: true })], spacing: { after: 80, line: 240 } }),
    ];
  }
  if (b.image) {
    const data = fs.readFileSync(b.image);
    return [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 20 }, children: [new ImageRun({ type: "png", data, transformation: { width: b.w, height: b.h } })] }),
      new Paragraph({ children: runs(b.caption, { size: SZ_SMALL }), alignment: AlignmentType.JUSTIFIED, spacing: { after: 80, line: 240 } }),
    ];
  }
  return [];
}
function box(title, blocks) {
  const children = [PB(title, { after: 60 })];
  for (const b of blocks) children.push(...block(b));
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: [W], rows: [new TableRow({ children: [cell(children, W)] })] });
}
function signatures(names) {
  const n = names.length; const cw = n === 2 ? [4300, 1038, 4300] : [W];
  const line = (t) => [new Paragraph({ spacing: { before: 480, after: 0 }, alignment: AlignmentType.CENTER, children: [new TextRun({ text: "______________________________", font: FONT, size: SZ })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: t, font: FONT, size: SZ })] })];
  const cells = n === 2 ? [cell(line(names[0]), cw[0], { borders: NOBOX }), cell([EMPTY()], cw[1], { borders: NOBOX }), cell(line(names[1]), cw[2], { borders: NOBOX })]
    : [cell(line(names[0]), W, { borders: NOBOX })];
  return new Table({ width: { size: W, type: WidthType.DXA }, columnWidths: cw, borders: NOBOX, rows: [new TableRow({ children: cells })] });
}
const dateLine = () => new Paragraph({ alignment: AlignmentType.LEFT, spacing: { before: 200, after: 0 }, children: [new TextRun({ text: C.dataLocal, font: FONT, size: SZ })] });
const pageBreak = () => new Paragraph({ children: [new PageBreak()] });

// ---- documento
const body = [];
// FOLHA 1
body.push(headerTable(), ...folhaTitle(1, "Não ultrapassar 3 (três) páginas de texto."), idTable(true), EMPTY());
for (const s of C.folha1) { body.push(box(s.titulo, s.blocos)); body.push(EMPTY()); }
body.push(signatures(["Assinatura do(a) Orientador(a)", "Assinatura do(a) Bolsista"]), dateLine(), pageBreak());
// FOLHA 2
body.push(headerTable(), ...folhaTitle(2, "Não ultrapassar 1 (uma) página de texto."), idTable(false), EMPTY(),
  box(C.folha2.titulo, C.folha2.blocos), signatures(["Assinatura do(a) Orientador(a)"]), dateLine(), pageBreak());
// FOLHA 3
body.push(headerTable(), ...folhaTitle(3, "Não ultrapassar 1 (uma) página de texto."), idTable(false), EMPTY(),
  box(C.folha3.titulo, C.folha3.blocos), signatures(["Assinatura do(a) Bolsista"]), dateLine());

const doc = new Document({
  creator: "LaCrio/FURG", title: "Relatório Final PIBIC/PIBITI CNPq — Matheus Oliveira (2025–2026)",
  styles: { default: { document: { run: { font: FONT, size: SZ } } } },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, right: 1134, bottom: 1693, left: 1134, footer: 1134, header: 0 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: SZ_SMALL })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(OUT, buf); console.log("salvo:", OUT, (buf.length / 1024).toFixed(0), "KB"); });
