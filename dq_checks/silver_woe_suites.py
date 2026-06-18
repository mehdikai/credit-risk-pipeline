"""GE expectation suite for STEP 2 (cont.) — SILVER WoE output (silver_woe).

Output contract: every WoE-encoded column must be numeric, non-null, and
within a plausible WoE range (generous +/-10 bound — observed values for
this dataset sit well within +/-1, the wide bound just guards against a
binning regression producing extreme/broken values).
"""
import great_expectations.expectations as gxe

WOE_VALUE_BOUND = 10


def silver_woe_expectations(woe_columns):
    expectations = [
        gxe.ExpectColumnValuesToNotBeNull(column="ID"),
        gxe.ExpectColumnValuesToBeUnique(column="ID"),
        gxe.ExpectColumnValuesToNotBeNull(column="TARGET"),
        gxe.ExpectColumnDistinctValuesToBeInSet(column="TARGET", value_set=[0, 1]),
    ]
    for col in woe_columns:
        expectations.append(gxe.ExpectColumnValuesToNotBeNull(column=col))
        expectations.append(
            gxe.ExpectColumnValuesToBeBetween(
                column=col, min_value=-WOE_VALUE_BOUND, max_value=WOE_VALUE_BOUND
            )
        )
    return expectations
