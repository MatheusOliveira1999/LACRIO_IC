# Relatório Final PIBIC/PIBITI CNPq (modelo FURG/PROPESP)

Gera `../Relatorio_Final_CNPq_FURG_2026.docx` no layout de `modelo_relatorio_final_cnpq_ic_it.doc`
(A4, Arial 10, espaço simples, Folha 1 ≤ 3 páginas de texto; Folhas 2 e 3 = 1 página).

- `conteudo_cnpq.js` — todo o texto (identificação, seções da Folha 1, Folha 2 do orientador, Folha 3 do bolsista).
  Versão 2 (18/09/2026): reescrita em linguagem simples, cada termo técnico explicado na primeira vez que aparece;
  a versão técnica original está em `conteudo_cnpq_v1_tecnico.js` (docx/pdf correspondentes com sufixo `_v1_tecnico`).
  Versão 2.1 (18/09): correções da auditoria `AUDITORIA_afirmacoes_referencias_2026-09-18.md` aplicadas (20,4%, preprint CrevasseSeg, clDice, MDE 22–31 cm, blocos/grupos, páginas das referências).
  Paginação alvo: texto da Folha 1 (Introdução → Perspectivas) nas páginas 1–3; Referências/Produção/assinaturas na página 4.
  Marcação: `**negrito**`, `*itálico*`, `^{sobrescrito}`; tabelas e figuras como objetos `{table|image}`.
- `build_cnpq.js` — layout (docx-js). Precisa do pacote npm `docx` (`npm install docx@8` numa pasta com `node_modules`).

```bash
cd relatorio_final/cnpq_build && NODE_PATH=/tmp/claude-1000/docxjs/node_modules node build_cnpq.js   # ou npm install docx@8 aqui
soffice --headless --convert-to pdf --outdir .. ../Relatorio_Final_CNPq_FURG_2026.docx
```
Campos a preencher: nº do processo, matrícula, área/subárea CNPq, produção técnico-científica, avaliação do orientador (Folha 2);
a autoavaliação (Folha 3) é um rascunho para revisão do bolsista.
