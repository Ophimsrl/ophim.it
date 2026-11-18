import math

import pytest
from mistralai import agents

from mistralai.vibe_agents.sdk.authoring.permissions import always_denies

from main import (
    _DATASETS,
    _DATASET_INFO,
    TOOL_POLICY,
    Aggregation,
    agent,
    current_dataset,
    describe_columns,
    filter_rows,
    group_by_aggregate,
    load_csv_content,
)

SAMPLE_CSV = """data,regione,citta,vendite,quantita
2026-01-05,Nord,Milano,120.5,3
2026-01-06,Sud,Napoli,80.0,2
2026-01-07,Nord,Torino,60.25,1
2026-01-08,Centro,Roma,200.0,5
2026-01-09,Sud,Bari,45.75,1
2026-01-10,Nord,Milano,99.5,2
"""


@pytest.fixture(autouse=True)
def clean_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MISTRAL_API_KEY", "test-dummy-key")
    _DATASETS.clear()
    _DATASET_INFO.clear()


def ctx() -> agents.ToolCallbackContext:
    return agents.ToolCallbackContext.local()


async def test_load_describe_filter_aggregate() -> None:
    info = await load_csv_content(ctx(), csv_text=SAMPLE_CSV, source="test")
    assert info["rows"] == 6
    assert {c["name"] for c in info["columns"]} == {"data", "regione", "citta", "vendite", "quantita"}
    assert len(info["sample_rows"]) == 5

    desc = await describe_columns(ctx())
    by_name = {c["name"]: c for c in desc["columns"]}
    assert "float" in by_name["vendite"]["dtype"]
    assert by_name["regione"]["unique_count"] == 3
    assert by_name["regione"]["null_count"] == 0
    assert by_name["vendite"]["min"] == 45.75

    filt = await filter_rows(ctx(), column="regione", operator="eq", value="Nord")
    assert filt["matched"] == 3
    assert filt["returned"] == 3
    assert filt["truncated"] is False

    agg = await group_by_aggregate(
        ctx(),
        group_by=["regione"],
        aggregations=[
            Aggregation(column="vendite", function="sum"),
            Aggregation(column="vendite", function="count"),
        ],
    )
    assert agg["groups"] == 3
    rows = {r["regione"]: r for r in agg["rows"]}
    assert math.isclose(rows["Nord"]["vendite__sum"], 280.25)
    assert rows["Nord"]["vendite__count"] == 3
    assert math.isclose(rows["Sud"]["vendite__sum"], 125.75)

    cur = await current_dataset(ctx())
    assert cur["rows"] == 6
    assert cur["source"] == "test"


async def test_filter_numeric_operator_and_truncation() -> None:
    await load_csv_content(ctx(), csv_text=SAMPLE_CSV, source="test")
    filt = await filter_rows(ctx(), column="vendite", operator="gt", value="100", limit=1)
    assert filt["matched"] == 2
    assert filt["returned"] == 1
    assert filt["truncated"] is True

    contained = await filter_rows(ctx(), column="citta", operator="contains", value="mil")
    assert contained["matched"] == 2


async def test_unknown_column_raises_tool_error() -> None:
    await load_csv_content(ctx(), csv_text=SAMPLE_CSV, source="test")
    with pytest.raises(agents.ToolError):
        await filter_rows(ctx(), column="inesistente", operator="eq", value="x")


async def test_no_dataset_raises_tool_error() -> None:
    with pytest.raises(agents.ToolError):
        await current_dataset(ctx())


def test_permission_policy_allows_only_fixed_tools() -> None:
    assert not always_denies(TOOL_POLICY, "csv.load_csv_from_drive")
    assert not always_denies(TOOL_POLICY, "csv.filter_rows")
    assert not always_denies(TOOL_POLICY, "connector_google_drive_mcp.read_file")
    assert always_denies(TOOL_POLICY, "policy.run_typescript")
    assert always_denies(TOOL_POLICY, "file_system.bash")
    assert always_denies(TOOL_POLICY, "process.start")
    assert always_denies(TOOL_POLICY, "client.anything_else")


async def test_scripted_model_turn() -> None:
    model = agents.testing.ScriptedModel(
        [
            agents.testing.call("load_csv_content", csv_text=SAMPLE_CSV, source="test"),
            agents.testing.call(
                "group_by_aggregate",
                group_by=["regione"],
                aggregations=[{"column": "vendite", "function": "sum"}],
            ),
            agents.testing.reply("Il Nord ha venduto 280.25 in totale."),
        ]
    )
    state = await agent.override(model=model).run("Quanto ha venduto ogni regione?")
    assert isinstance(state, agents.Completed), state
    assert [c.name for c in state.tool_calls] == ["csv.load_csv_content", "csv.group_by_aggregate"]
    assert state.text == "Il Nord ha venduto 280.25 in totale."
    model.assert_consumed()
