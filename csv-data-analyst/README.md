# csv-data-analyst

Agente analista di dati conversazionale costruito con il Mistral Agents SDK (`mistralai-agents`).
L'utente fornisce un link a un file CSV su Google Drive e pone domande in linguaggio naturale;
l'agente risponde con numeri reali presi dai dati.

## Come funziona

- Il file viene recuperato con il connettore Mistral `google_drive_mcp`, che agisce con la
  connessione Google dell'utente che avvia la sessione (on-behalf-of), quindi il deployment è hardened.
- I dati passano solo attraverso tool fissi del namespace `csv`, chiamati direttamente dal modello
  (`model_access="direct"`):
  - `csv.load_csv_from_drive` — legge il CSV da Drive (link, ID o nome da cercare) e lo carica in sessione
  - `csv.load_csv_content` — fallback con il contenuto testuale del CSV
  - `csv.current_dataset` — riepilogo del dataset caricato in sessione
  - `csv.describe_columns` — tipi, nulli, cardinalità, min/max/media, valori frequenti
  - `csv.filter_rows` — filtri su una colonna (eq, ne, gt, ge, lt, le, contains, in, is_null, ...)
  - `csv.group_by_aggregate` — aggregati group-by (sum, mean, median, min, max, count, nunique, std)
- Nessuna esecuzione di codice arbitraria: nessun sandbox (NoSandbox) e una `permission_policy` con
  `default="deny"` che permette solo `csv.*` e i tool del connettore, negando `policy.run_typescript`
  (il tool di esecuzione codice) e ogni altro built-in: il modello può chiamare solo i tool fissi.
- Le sessioni del deployment sono persistenti (Mistral le salva): i follow-up riutilizzano la stessa
  conversazione e il dataset già caricato, senza che l'utente debba rispiegare nulla.

## Sviluppo locale

```bash
uv sync
uv run pytest -o asyncio_mode=auto   # test offline con ScriptedModel, nessuna API key richiesta
```

Con una `MISTRAL_API_KEY` in `.env`:

```bash
uv run --env-file .env agents chat        # chat locale
uv run --env-file .env agents info        # verifica entrypoint e capability
```

## Deploy su Mistral Cloud (gestito)

Il progetto viene pushato su un repository GitHub; il deployment managed lo builda da quel commit:

```bash
git add --all && git commit -m "csv-data-analyst agent" && git push
# deployment hardened (OBO per il connettore Drive), es. via Mistral MCP:
# mistral_mcp_deploy_workflow_deployment(deployment_name="csv-data-analyst",
#   github_url="https://github.com/Ophimsrl/ophim.it", revision="<commit-sha>",
#   build_directory="csv-data-analyst", dockerfile_path="Dockerfile", hardened=true)
```

## Usare l'agente deployato

- In Vibe Code Web: il link `https://chat.mistral.ai/managed-agents/csv-data-analyst` (lo stampa anche `agents deploy`).
- Da terminale: `uv run --env-file .env agents chat csv-data-analyst`
- Da codice: `agents.Agent.load_from(agent_name="csv-data-analyst")`, poi `start_session()` e `session.run(...)`
  per conversazioni multi-turno; le sessioni restano recuperabili per `session_id`.
- In Mistral Studio: Managed Agents, sessioni e tracce dei turn.

Prerequisito: l'utente deve aver collegato il connettore `google_drive_mcp` al proprio account Google
in Mistral Studio (console.mistral.ai/build/connectors).
