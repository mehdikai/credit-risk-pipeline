"""GE expectation suites for STEP 2 — SILVER (clean / join / dedup / label).

Two contracts are validated:
  - input contract (silver_input_*): assumptions the cleaning transform
    relies on, checked on the raw Bronze application_record before any
    transformation. Uses `mostly=` thresholds since some "imperfections"
    here (e.g. OCCUPATION_TYPE nulls, a handful of duplicate IDs) are
    expected and handled by the transform itself.
  - output contract (silver_output_*): guarantees the cleaned
    silver_clean table makes to downstream consumers (Week 4 WoE step).
    Strict / zero-tolerance.
"""
import great_expectations.expectations as gxe


def silver_input_expectations():
    return [
        gxe.ExpectColumnValuesToNotBeNull(column="ID"),
        # ~47 duplicate IDs (94/438557 rows) are a known characteristic of
        # this raw dataset, handled by dedup in the transform below.
        gxe.ExpectColumnValuesToBeUnique(column="ID", mostly=0.999),
        # OCCUPATION_TYPE is ~30.6% null in the raw data (unemployed /
        # pensioners) — handled via "Unknown" imputation in the transform.
        gxe.ExpectColumnValuesToNotBeNull(column="OCCUPATION_TYPE", mostly=0.60),
        gxe.ExpectColumnValuesToNotBeNull(column="AMT_INCOME_TOTAL"),
        gxe.ExpectColumnValuesToNotBeNull(column="DAYS_BIRTH"),
        gxe.ExpectColumnValuesToNotBeNull(column="DAYS_EMPLOYED"),
    ]


def silver_output_expectations():
    return [
        gxe.ExpectColumnValuesToNotBeNull(column="ID"),
        gxe.ExpectColumnValuesToBeUnique(column="ID"),
        gxe.ExpectColumnValuesToNotBeNull(column="TARGET"),
        gxe.ExpectColumnDistinctValuesToBeInSet(column="TARGET", value_set=[0, 1]),
        gxe.ExpectColumnValuesToNotBeNull(column="OCCUPATION_TYPE"),
        gxe.ExpectColumnValuesToNotBeNull(column="AGE_YEARS"),
        gxe.ExpectColumnValuesToBeBetween(column="AGE_YEARS", min_value=18, max_value=100),
        gxe.ExpectColumnValuesToNotBeNull(column="EMPLOYED_YEARS"),
        gxe.ExpectColumnValuesToNotBeNull(column="IS_UNEMPLOYED"),
        gxe.ExpectColumnDistinctValuesToBeInSet(column="IS_UNEMPLOYED", value_set=[0, 1]),
        gxe.ExpectColumnValuesToNotBeNull(column="EMPLOYED_RATIO"),
        gxe.ExpectColumnValuesToNotBeNull(column="INCOME_PER_FAM_MEMBER"),
        gxe.ExpectTableRowCountToBeBetween(min_value=1),
    ]
