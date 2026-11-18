import io
import math
import re
from datetime import datetime
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from mistralai import agents

DRIVE_CONNECTOR = "google_drive_mcp"
MAX_CSV_CHARS = 8_000_000
MAX_RETURN_ROWS = 100
SAMPLE_ROWS = 5

Operator = Literal[
    "eq", "ne", "gt", "ge", "lt", "le",
    "contains", "not_contains", "in", "not_in",
    "is_null", "not_null",
]
AggFunction = Literal["sum", "mean", "median", "min", "max", "count", "nunique", "std"]


class Aggregation(BaseModel):
    column: str = Field(description="Colonna su cui calcolare l'aggregazione.")
    function: AggFunction = Field(description="Funzione di aggregazione.")


_DATASETS: dict[str, pd.DataFrame] = {}
_DATASET_INFO: dict[str, dict[str, Any]] = {}

_DRIVE_READ_TOOLS = (
    "read_file", "download_file", "get_file_content",
    "read_file_content", "export_file", "get_file",
)
_DRIVE_SEARCH_TOOLS = ("search_files", "search", "list_files", "search_drive", "find_files")


def _session_key(ctx: agents.ToolCallbackContext) -> str:
    return ctx.session.id


def _get_dataset(ctx: agents.ToolCallbackContext) -> pd.DataFrame:
    df = _DATASETS.get(_session_key(ctx))
    if df is None:
        raise agents.ToolError(
            "Nessun dataset caricato in questa sessione. Carica il file con csv.load_csv_from_drive "
            "passando il link Google Drive già menzionato nella conversazione."
        )
    return df


def _clean_value(value: Any) -> Any:
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: _clean_value(value) for key, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _extract_file_id(ref: str) -> str | None:
    ref = ref.strip()
    match = (
        re.search(r"/d/([a-zA-Z0-9_-]+)", ref)
        or re.search(r"[?&]id=([a-zA-Z0-9_-]+)", ref)
    )
    if match:
        return match.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", ref):
        return ref
    return None


async def _drive_call(tool_name: str, arguments: dict[str, Any]) -> str:
    from mistralai.workflows.client import get_mistral_client
    from mistralai.workflows.core.auth import RunAs

    client = get_mistral_client(run_as=RunAs.EXECUTOR)
    response = await client.beta.connectors.call_tool_async(
        connector_id_or_name=DRIVE_CONNECTOR,
        tool_name=tool_name,
        arguments=arguments,
    )
    if getattr(response, "is_error", False):
        texts = [b.text for b in response.content if getattr(b, "text", None)]
        raise RuntimeError("\n".join(texts) or "connector error")
    return "\n".join(b.text for b in response.content if getattr(b, "text", None))


async def _drive_call_first_ok(candidates: tuple[str, ...], arg_variants: list[dict[str, Any]]) -> str:
    errors: list[str] = []
    for tool in candidates:
        for args in arg_variants:
            try:
                out = await _drive_call(tool, args)
                if out.strip():
                    return out
            except Exception as exc:
                errors.append(f"{tool}: {type(exc).__name__}")
    raise RuntimeError(
        "Nessun tool del connettore google_drive_mcp ha funzionato: " + "; ".join(errors[:6])
    )


async def _fetch_csv_text(ref: str) -> tuple[str | None, str]:
    file_id = _extract_file_id(ref)
    if file_id is None:
        search = await _drive_call_first_ok(
            _DRIVE_SEARCH_TOOLS,
            [{"query": ref}, {"q": ref}, {"name": ref, "mimeType": "text/csv"}],
        )
        ids = re.findall(r"\b([a-zA-Z0-9_-]{25,})\b", search)
        if not ids:
            raise agents.ToolError(
                "Non ho trovato un ID file nei risultati di ricerca di Drive. "
                "Passa il link completo del file (https://drive.google.com/file/d/<ID>/...). "
                f"Risultati: {search[:400]}"
            )
        file_id = ids[0]
    text = await _drive_call_first_ok(
        _DRIVE_READ_TOOLS,
        [{"file_id": file_id}, {"fileId": file_id}, {"id": file_id}],
    )
    return file_id, text


def _parse_csv(text: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(io.StringIO(text))
        except Exception as exc:
            raise agents.ToolError(f"Il file non sembra un CSV valido: {type(exc).__name__}.") from exc
    if len(df.columns) == 0 or len(df) == 0:
        raise agents.ToolError("Il CSV è vuoto o non ha colonne leggibili.")
    return df


def _dataset_info(df: pd.DataFrame, source: str, file_id: str | None) -> dict[str, Any]:
    return {
        "source": source,
        "file_id": file_id,
        "rows": int(len(df)),
        "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        "sample_rows": _records(df.head(SAMPLE_ROWS)),
    }


def _store_dataset(ctx: agents.ToolCallbackContext, df: pd.DataFrame, source: str, file_id: str | None) -> dict[str, Any]:
    key = _session_key(ctx)
    _DATASETS[key] = df
    info = _dataset_info(df, source, file_id)
    _DATASET_INFO[key] = info
    return info


@agents.tool(model_access="direct", namespace="csv", annotations=agents.ToolAnnotations(read_only_hint=True))
async def load_csv_from_drive(
    ctx: agents.ToolCallbackContext, /,
    file_url_or_id: Annotated[str, Field(description="Link Google Drive del CSV (https://drive.google.com/file/d/<ID>/...), oppure ID del file, oppure nome da cercare in Drive.")],
) -> dict[str, Any]:
    """Carica un file CSV dal Google Drive dell'utente tramite il connettore google_drive_mcp e lo memorizza per questa sessione. Restituisce righe, colonne, tipi e un campione."""
    try:
        file_id, text = await _fetch_csv_text(file_url_or_id)
    except agents.ToolError:
        raise
    except Exception as exc:
        raise agents.ToolError(
            "Non sono riuscito a leggere il file da Google Drive "
            f"({type(exc).__name__}). Verifica che il connettore google_drive_mcp sia collegato al tuo "
            "account in Mistral Studio (console.mistral.ai/build/connectors) e che il file sia accessibile "
            "da quell'account, poi riprova. In alternativa, leggi il file con i tool del connettore "
            "google_drive_mcp e passa il contenuto a csv.load_csv_content."
        ) from exc
    if len(text) > MAX_CSV_CHARS:
        raise agents.ToolError(
            f"Il file supera il limite di {MAX_CSV_CHARS} caratteri ({len(text)}). Usa un CSV più piccolo."
        )
    df = _parse_csv(text)
    return _store_dataset(ctx, df, source=f"google_drive:{file_id}", file_id=file_id)


@agents.tool(model_access="direct", namespace="csv", annotations=agents.ToolAnnotations(read_only_hint=True))
async def load_csv_content(
    ctx: agents.ToolCallbackContext, /,
    csv_text: Annotated[str, Field(description="Contenuto testuale completo del file CSV.")],
    source: Annotated[str, Field(description="Origine del dato, es. nome del file.")] = "incollato",
) -> dict[str, Any]:
    """Carica il contenuto testuale di un CSV e lo memorizza per questa sessione (fallback quando il link Drive non è leggibile direttamente). Restituisce righe, colonne, tipi e un campione."""
    if len(csv_text) > MAX_CSV_CHARS:
        raise agents.ToolError(
            f"Il testo supera il limite di {MAX_CSV_CHARS} caratteri ({len(csv_text)}). Usa un CSV più piccolo."
        )
    df = _parse_csv(csv_text)
    return _store_dataset(ctx, df, source=source, file_id=None)


@agents.tool(model_access="direct", namespace="csv", annotations=agents.ToolAnnotations(read_only_hint=True))
async def current_dataset(ctx: agents.ToolCallbackContext, /) -> dict[str, Any]:
    """Restituisce quale dataset è attualmente caricato in questa sessione: origine, file ID, numero di righe, colonne e campione. Usalo nelle domande di follow-up per sapere cosa è già in memoria."""
    key = _session_key(ctx)
    info = _DATASET_INFO.get(key)
    if info is None:
        raise agents.ToolError(
            "Nessun dataset caricato in questa sessione. Carica il file con csv.load_csv_from_drive "
            "passando il link Google Drive già menzionato nella conversazione."
        )
    return info


@agents.tool(model_access="direct", namespace="csv", annotations=agents.ToolAnnotations(read_only_hint=True))
async def describe_columns(
    ctx: agents.ToolCallbackContext, /,
    columns: Annotated[list[str] | None, Field(description="Colonne da descrivere; se omesso, tutte le colonne.")] = None,
) -> dict[str, Any]:
    """Descrive le colonne del dataset caricato: tipo, valori nulli, cardinalità, min/max/media per le numeriche, valori più frequenti per le testuali, esempi."""
    df = _get_dataset(ctx)
    selected = columns if columns else list(df.columns)
    missing = [c for c in selected if c not in df.columns]
    if missing:
        raise agents.ToolError(
            f"Colonne non presenti: {missing}. Disponibili: {list(df.columns)}"
        )
    out: list[dict[str, Any]] = []
    for name in selected:
        series = df[name]
        entry: dict[str, Any] = {
            "name": name,
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
            "sample_values": [_clean_value(v) for v in series.dropna().unique()[:5]],
        }
        if pd.api.types.is_numeric_dtype(series):
            entry["min"] = _clean_value(series.min())
            entry["max"] = _clean_value(series.max())
            entry["mean"] = _clean_value(series.mean())
        elif pd.api.types.is_datetime64_any_dtype(series):
            entry["min"] = _clean_value(series.min())
            entry["max"] = _clean_value(series.max())
        else:
            top = series.value_counts(dropna=True).head(5)
            entry["top_values"] = [
                {"value": _clean_value(k), "count": int(v)} for k, v in top.items()
            ]
        out.append(entry)
    return {"rows": int(len(df)), "columns": out}


def _coerce_value(series: pd.Series, value: Any) -> Any:
    if value is None or isinstance(value, list):
        return value
    if pd.api.types.is_numeric_dtype(series):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(value, errors="coerce")
    if isinstance(value, str) and pd.api.types.is_bool_dtype(series):
        return value.strip().lower() in ("true", "1", "si", "sì", "yes")
    return value


def _apply_operator(series: pd.Series, operator: Operator, value: Any) -> pd.Series:
    if operator == "is_null":
        return series.isna()
    if operator == "not_null":
        return series.notna()
    if operator in ("in", "not_in"):
        values = value if isinstance(value, list) else [value]
        mask = series.isin([_coerce_value(series, v) for v in values])
        return mask if operator == "in" else ~mask
    value = _coerce_value(series, value)
    if operator == "eq":
        return series == value
    if operator == "ne":
        return series != value
    if operator == "gt":
        return series > value
    if operator == "ge":
        return series >= value
    if operator == "lt":
        return series < value
    if operator == "le":
        return series <= value
    text = series.astype(str)
    if operator == "contains":
        return text.str.contains(str(value), case=False, na=False)
    if operator == "not_contains":
        return ~text.str.contains(str(value), case=False, na=False)
    raise agents.ToolError(f"Operatore non supportato: {operator}")


@agents.tool(model_access="direct", namespace="csv", annotations=agents.ToolAnnotations(read_only_hint=True))
async def filter_rows(
    ctx: agents.ToolCallbackContext, /,
    column: Annotated[str, Field(description="Colonna su cui filtrare.")],
    operator: Annotated[Operator, Field(description="Operatore di confronto. Per 'in'/'not_in' passa una lista in 'value'.")],
    value: Annotated[Any, Field(description="Valore di confronto (lista per 'in'/'not_in'; ignorato per 'is_null'/'not_null').")] = None,
    limit: Annotated[int, Field(description=f"Numero massimo di righe restituite (max {MAX_RETURN_ROWS}).")] = 50,
) -> dict[str, Any]:
    """Filtra le righe del dataset caricato con un confronto su una colonna e restituisce le righe corrispondenti (limitate) più il conteggio totale."""
    df = _get_dataset(ctx)
    if column not in df.columns:
        raise agents.ToolError(f"Colonna '{column}' non presente. Disponibili: {list(df.columns)}")
    mask = _apply_operator(df[column], operator, value)
    matched = df[mask]
    limit = max(1, min(limit, MAX_RETURN_ROWS))
    shown = matched.head(limit)
    return {
        "column": column,
        "operator": operator,
        "value": _clean_value(value),
        "matched": int(len(matched)),
        "returned": int(len(shown)),
        "truncated": bool(len(matched) > len(shown)),
        "rows": _records(shown),
    }


@agents.tool(model_access="direct", namespace="csv", annotations=agents.ToolAnnotations(read_only_hint=True))
async def group_by_aggregate(
    ctx: agents.ToolCallbackContext, /,
    group_by: Annotated[list[str], Field(description="Colonne di raggruppamento (una o più).")],
    aggregations: Annotated[list[Aggregation], Field(description="Aggregazioni da calcolare per ogni gruppo.")],
    limit: Annotated[int, Field(description=f"Numero massimo di gruppi restituiti (max {MAX_RETURN_ROWS}).")] = 100,
) -> dict[str, Any]:
    """Calcola aggregati group-by sul dataset caricato: per ogni combinazione delle colonne di gruppo restituisce le aggregazioni richieste (sum, mean, median, min, max, count, nunique, std)."""
    df = _get_dataset(ctx)
    if not group_by:
        raise agents.ToolError("group_by deve contenere almeno una colonna.")
    missing = [c for c in group_by if c not in df.columns]
    if missing:
        raise agents.ToolError(f"Colonne di gruppo non presenti: {missing}. Disponibili: {list(df.columns)}")
    if not aggregations:
        raise agents.ToolError("aggregations deve contenere almeno un'aggregazione.")
    specs: dict[str, tuple[str, str]] = {}
    resolved: list[dict[str, str]] = []
    for agg in aggregations:
        column = agg.column if isinstance(agg, Aggregation) else agg["column"]
        function = agg.function if isinstance(agg, Aggregation) else agg["function"]
        if column not in df.columns:
            raise agents.ToolError(f"Colonna '{column}' non presente. Disponibili: {list(df.columns)}")
        specs[f"{column}__{function}"] = (column, function)
        resolved.append({"column": column, "function": function})
    grouped = df.groupby(group_by, dropna=False).agg(**specs).reset_index()
    limit = max(1, min(limit, MAX_RETURN_ROWS))
    shown = grouped.head(limit)
    return {
        "group_by": group_by,
        "aggregations": resolved,
        "groups": int(len(grouped)),
        "returned": int(len(shown)),
        "truncated": bool(len(grouped) > len(shown)),
        "rows": _records(shown),
    }


INSTRUCTIONS = """Sei un analista di dati conversazionale. L'utente ti dà un link a un file CSV nel suo Google Drive e ti fa domande in linguaggio naturale sui dati.

Regole:
- Per leggere il file usa SEMPRE il tool csv.load_csv_from_drive passando il link o l'ID del file: usa il connettore google_drive_mcp con la connessione Google dell'utente che ha avviato la sessione.
- Per esplorare e rispondere usa SOLO i tool fissi del namespace csv: csv.load_csv_from_drive, csv.load_csv_content, csv.current_dataset, csv.describe_columns, csv.filter_rows, csv.group_by_aggregate. Non hai accesso a codice arbitrario, shell o file: non tentare di eseguirne.
- Non inventare mai numeri: ogni cifra, conteggio o tabella nella risposta deve venire dal risultato di un tool csv.*. Se non hai i dati, caricali o filtrali prima di rispondere.
- Prima di rispondere a domande quantitative, conosci le colonne con csv.describe_columns (o dal riepilogo di csv.load_csv_from_drive / csv.current_dataset).
- La sessione è persistente: ricorda i messaggi precedenti e il dataset caricato. Nelle domande di follow-up riusa il dataset già caricato (verifica con csv.current_dataset) e non chiedere di nuovo il link se è già in conversazione. Se un tool segnala che il dataset non è più in memoria, ricarica il file con csv.load_csv_from_drive usando lo stesso link della conversazione.
- Rispondi in italiano, in modo conciso e diretto. Riporta i numeri reali prodotti dai tool, indica filtri e gruppi usati, e se il risultato è troncato (campo "truncated": true) dillo e proponi di affinare il filtro o il raggruppamento.
- Se il file non è leggibile o il connettore non risulta collegato, spiega che bisogna collegare google_drive_mcp al proprio account in Mistral Studio (console.mistral.ai/build/connectors) e riprovare.
"""

TOOL_POLICY = agents.PermissionPolicy(
    default="deny",
    rules=(
        agents.PermissionRule(
            tools=("csv.*", "connectors.*", "connector_*"),
            decision="allow",
        ),
        agents.PermissionRule(
            tools=("policy.*", "self.*", "file_system.*", "process.*", "agents.*", "subagent.*", "skill.*"),
            decision="deny",
        ),
    ),
)

agent = agents.Agent(
    model="mistral-large-4",
    name="csv-data-analyst",
    description=(
        "Analista di dati conversazionale su file CSV in Google Drive. Tool fissi per caricare il file, "
        "descrivere le colonne, filtrare le righe e calcolare aggregati group-by, senza esecuzione di codice arbitrario."
    ),
    instructions=INSTRUCTIONS,
    tools=[
        load_csv_from_drive,
        load_csv_content,
        current_dataset,
        describe_columns,
        filter_rows,
        group_by_aggregate,
    ],
    connectors=[agents.connector(DRIVE_CONNECTOR)],
    builtins=agents.Builtins(skills=False, subagents=False),
    permission_policy=TOOL_POLICY,
)
