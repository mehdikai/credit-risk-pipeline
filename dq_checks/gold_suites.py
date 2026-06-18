"""GE expectation suite for STEP 3 — GOLD (final ML-ready feature table).

Strict output contract: gold_features must contain exactly the WoE feature
columns + TARGET (nothing else — no ID, no raw/helper columns), with zero
nulls anywhere and TARGET in {0,1}.
"""
import great_expectations.expectations as gxe


def gold_expectations(feature_columns):
    expectations = [
        gxe.ExpectTableColumnsToMatchSet(
            column_set=feature_columns + ["TARGET"], exact_match=True
        ),
        gxe.ExpectColumnValuesToNotBeNull(column="TARGET"),
        gxe.ExpectColumnDistinctValuesToBeInSet(column="TARGET", value_set=[0, 1]),
    ]
    for col in feature_columns:
        expectations.append(gxe.ExpectColumnValuesToNotBeNull(column=col))
    return expectations
