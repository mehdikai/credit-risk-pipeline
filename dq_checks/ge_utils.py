"""Great Expectations runner backed by a persistent context + real Data Docs.

Every validation runs through a GE Checkpoint with an UpdateDataDocsAction,
so each call regenerates the official GE Data Docs HTML site (full history
of every run, per-expectation drill-down, charts, search/navigation) at
DATA_DOCS_INDEX. This replaces the earlier lightweight custom HTML
renderer — visual polish now comes from GE itself rather than a hand-rolled
template.

The context and all suites/checkpoints are persisted under dq_checks/gx/
(regenerated on every run via add_or_update — safe to delete and rebuild).
"""
from pathlib import Path

import great_expectations as gx
from great_expectations.checkpoint.actions import UpdateDataDocsAction

GX_PROJECT_ROOT = Path("dq_checks")
DATA_DOCS_INDEX = GX_PROJECT_ROOT / "gx" / "uncommitted" / "data_docs" / "local_site" / "index.html"

_context = None


def _get_context():
    global _context
    if _context is None:
        _context = gx.get_context(mode="file", project_root_dir=str(GX_PROJECT_ROOT))
    return _context


def _get_or_add_datasource(context, name):
    try:
        return context.data_sources.get(name)
    except KeyError:
        return context.data_sources.add_pandas(name)


def _get_or_add_asset(datasource, name):
    try:
        return datasource.get_asset(name)
    except LookupError:
        return datasource.add_dataframe_asset(name=name)


def _get_or_add_batch_definition(asset, name):
    try:
        return asset.get_batch_definition(name)
    except KeyError:
        return asset.add_batch_definition_whole_dataframe(name)


def validate_dataframe(df, suite_name, expectations):
    """Run a list of GE expectations against a pandas dataframe via a
    Checkpoint, updating the real GE Data Docs site as a side effect.

    Returns the validation result as a dict (success flag, statistics,
    per-expectation results) — same shape every caller already expects,
    regardless of the underlying GE plumbing.
    """
    context = _get_context()

    datasource = _get_or_add_datasource(context, f"pandas_{suite_name}")
    asset = _get_or_add_asset(datasource, suite_name)
    batch_definition = _get_or_add_batch_definition(asset, "batch")

    # add_or_update with a *fresh* ExpectationSuite replaces any previously
    # persisted expectation list — re-running always reflects exactly the
    # current suite definition in code, never a stale accumulation.
    suite = context.suites.add_or_update(gx.ExpectationSuite(name=f"{suite_name}_suite"))
    for expectation in expectations:
        suite.add_expectation(expectation)

    validation_definition = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(
            name=f"{suite_name}_validation", data=batch_definition, suite=suite
        )
    )
    checkpoint = context.checkpoints.add_or_update(
        gx.Checkpoint(
            name=f"{suite_name}_checkpoint",
            validation_definitions=[validation_definition],
            actions=[UpdateDataDocsAction(name="update_data_docs")],
        )
    )

    checkpoint_result = checkpoint.run(batch_parameters={"dataframe": df})
    validation_result = next(iter(checkpoint_result.run_results.values()))
    return validation_result.describe_dict()
