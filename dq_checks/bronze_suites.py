"""GE expectation suites for STEP 1 — BRONZE (raw CSV checks).

Each suite is split into:
  - critical: schema / identity checks. A failure here aborts ingestion.
  - warn: soft data-quality checks. A failure here is logged but does not
    block the Bronze write (raw data is kept as-is).
"""
import great_expectations.expectations as gxe

APPLICATION_RECORD_COLUMNS = [
    "ID", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
    "AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED", "FLAG_MOBIL", "FLAG_WORK_PHONE",
    "FLAG_PHONE", "FLAG_EMAIL", "OCCUPATION_TYPE", "CNT_FAM_MEMBERS",
]

CREDIT_RECORD_COLUMNS = ["ID", "MONTHS_BALANCE", "STATUS"]

CREDIT_RISK_DATASET_COLUMNS = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_status",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
]


def _columns_exist(columns):
    return [gxe.ExpectColumnToExist(column=c) for c in columns]


def application_record_expectations():
    # NOTE: ID uniqueness is NOT a Bronze blocker — this raw dataset is known
    # to contain ~47 duplicate IDs. Bronze keeps data untouched; dedup happens
    # in Silver (Week 3). We still flag it here as a soft warning.
    critical = _columns_exist(APPLICATION_RECORD_COLUMNS) + [
        gxe.ExpectColumnValuesToNotBeNull(column="ID"),
    ]
    warn = [
        gxe.ExpectColumnValuesToBeUnique(column="ID"),
        gxe.ExpectColumnValuesToNotBeNull(column="AMT_INCOME_TOTAL"),
        gxe.ExpectColumnDistinctValuesToBeInSet(column="CODE_GENDER", value_set=["M", "F"]),
    ]
    return critical, warn


def credit_record_expectations():
    critical = _columns_exist(CREDIT_RECORD_COLUMNS) + [
        gxe.ExpectColumnValuesToNotBeNull(column="ID"),
        gxe.ExpectColumnValuesToNotBeNull(column="MONTHS_BALANCE"),
    ]
    warn = [
        gxe.ExpectColumnDistinctValuesToBeInSet(
            column="STATUS", value_set=["0", "1", "2", "3", "4", "5", "C", "X"]
        ),
    ]
    return critical, warn


def credit_risk_dataset_expectations():
    critical = _columns_exist(CREDIT_RISK_DATASET_COLUMNS) + [
        gxe.ExpectColumnValuesToNotBeNull(column="person_age"),
    ]
    warn = [
        gxe.ExpectColumnDistinctValuesToBeInSet(column="loan_status", value_set=[0, 1]),
    ]
    return critical, warn


SUITES = {
    "application_record": application_record_expectations,
    "credit_record": credit_record_expectations,
    "credit_risk_dataset": credit_risk_dataset_expectations,
}
