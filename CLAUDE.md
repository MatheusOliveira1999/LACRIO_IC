# Instruções para o agente — LACRIO IC

Este arquivo é lido automaticamente ao iniciar cada sessão. Siga o protocolo abaixo.

---

## Ao abrir uma nova conversa

Execute estas etapas **antes de responder qualquer pergunta técnica**:

### 1. Leia o estado atual do projeto no Obsidian

Caminho do vault: `/home/matheus/Documents/obsidian/LACRIO/`

Leia nesta ordem:

| Arquivo | O que contém |
|---------|-------------|
| `Progresso.md` | Status global, marcos concluídos, linha do tempo, o que falta |
| `Resultados.md` | Métricas mais recentes por ano e por experimento |
| `Melhorias.md` | Lista de prioridades atual (P0, P1, P2) |

### 2. Verifique o estado do repositório

```bash
git log --oneline -5
git status
```

### 3. Leia os arquivos de memória

Verifique `/home/matheus/.claude/projects/-home-matheus-Documents-GitHub-LACRIO-IC/memory/MEMORY.md`
e os arquivos referenciados nele.

### 4. Apresente um resumo de contexto curto

Antes de perguntar o que o usuário quer fazer, diga em 3–5 linhas:
- Fase atual do projeto
- Último experimento / resultado registrado
- Próxima prioridade da lista

---

## Ao ouvir "finalizando" (ou equivalente)

Quando o usuário disser que está encerrando a sessão, execute **todas** as etapas abaixo:

### 1. Atualize `Progresso.md`

- Adicione os eventos de hoje na seção `### Abril 2026` (ou o mês corrente)
- Use o formato `- **DD/MM** — descrição concisa`
- Atualize o campo `**Fase:**` se mudou
- Atualize `**Progresso estimado:**` se avançou
- Atualize a lista `## O que falta (priorizado)` com o estado atual

### 2. Atualize `Resultados.md`

- Se houve novo treino ou validação, adicione seção `### 🔵 U-Net — [descrição] (AAAA-MM-DD)`
- Inclua tabela de métricas por ano (F1, P, R, IoU)
- Registre qual checkpoint foi salvo e em qual época

### 3. Atualize `Melhorias.md`

- Marque como ✅ qualquer etapa concluída
- Adicione novas etapas se foram identificadas

### 4. Salve memórias relevantes

Se houver decisões técnicas novas, preferências do usuário reveladas, ou aprendizados
que não estão no código, salve em:
`/home/matheus/.claude/projects/-home-matheus-Documents-GitHub-LACRIO-IC/memory/`

Use os tipos: `feedback` (preferências/correções), `project` (decisões técnicas),
`user` (perfil do usuário).

### 5. Confirme ao usuário

Responda com uma lista concisa do que foi atualizado, ex:
> Vault atualizado: Progresso.md (linha do tempo + prioridades), Resultados.md (Exp. C).
> Próxima sessão: validar Exp. C + active learning 2017–2019.

---

## Contexto permanente do projeto

- **Projeto:** Segmentação semântica de feições supraglaciais no Glaciar Schiaparelli
- **Dataset:** ~22k tiles RGB 512×512, resolução ~5.4 cm/px (drone)
- **Features:** `lakes`, `crevasses`, `channels`
- **Modelo atual:** U-Net ResNet34 (`03b_train_unet.py` + `04b_inference_unet.py`)
- **Ambiente conda:** `sam_glaciar`
- **Anos válidos para treino:** 2016, 2017, 2018, 2019 (2020 descartado — mosaico defeituoso)
- **Anotações limpas:** 50 positivas reais (após remoção de 120 máscaras vazias)
- **Meta F1:** 85–90% para lakes (edital CNPq)

## Regras de conduta

- Não use `conda run -n sam_glaciar` — o usuário já ativa o ambiente antes
- Respostas curtas e diretas; sem emojis
- Ao sugerir comandos, não inclua o prefixo do ambiente
- Ao referenciar arquivos do projeto, use links markdown relativos
