"""Lightweight Great Expectations runner used across all lakehouse layers."""
import json
from datetime import datetime, timezone
from pathlib import Path

import great_expectations as gx

from config.config import DQ_REPORT_PATH


def validate_dataframe(df, suite_name, expectations):
    """Run a list of GE expectations against a pandas dataframe.

    Returns the GE validation result as a dict (success flag, statistics,
    per-expectation results).
    """
    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas(f"pandas_{suite_name}")
    asset = data_source.add_dataframe_asset(name=suite_name)
    batch_definition = asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
    for expectation in expectations:
        suite.add_expectation(expectation)

    result = batch.validate(suite)
    return result.describe_dict()


def write_report(table_name, report_name, result, extra=None):
    """Write a JSON + HTML DQ report for one table/check to DQ_REPORT_PATH."""
    report_dir = Path(DQ_REPORT_PATH)
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "table": table_name,
        "report": report_name,
        "timestamp": timestamp,
        "success": result["success"],
        "statistics": result["statistics"],
        "expectations": result["expectations"],
    }
    if extra:
        payload.update(extra)

    base = f"{report_name}_{table_name}_{timestamp}"
    json_path = report_dir / f"{base}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    html_path = report_dir / f"{base}.html"
    html_path.write_text(_render_html(payload))

    return json_path, html_path


def _render_html(payload):
    rows = "".join(
        "<tr><td>{}</td><td>{}</td><td style='color:{}'>{}</td></tr>".format(
            e["expectation_type"],
            e["kwargs"],
            "green" if e["success"] else "red",
            "PASS" if e["success"] else "FAIL",
        )
        for e in payload["expectations"]
    )
    status = "PASS" if payload["success"] else "FAIL"
    color = "green" if payload["success"] else "red"
    return f"""<html><head><title>DQ Report - {payload['table']}</title></head>
<body>
<h2>{payload['report']} — {payload['table']}</h2>
<p>Run: {payload['timestamp']}</p>
<p>Overall: <b style="color:{color}">{status}</b></p>
<p>{payload['statistics']}</p>
<table border="1" cellpadding="5">
<tr><th>Expectation</th><th>Kwargs</th><th>Result</th></tr>
{rows}
</table>
</body></html>"""
