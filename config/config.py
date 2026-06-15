# config/config.py
STORAGE_MODE = "local"  # switch to "minio" later

PATHS = {
    "local": {
        "bronze": "lakehouse/bronze/",
        "silver": "lakehouse/silver/",
        "gold":   "lakehouse/gold/",
    },
    "minio": {
        "bronze": "s3://bronze/",
        "silver": "s3://silver/",
        "gold":   "s3://gold/",
    }
}

BRONZE_PATH = PATHS[STORAGE_MODE]["bronze"]
SILVER_PATH = PATHS[STORAGE_MODE]["silver"]
GOLD_PATH   = PATHS[STORAGE_MODE]["gold"]

DQ_REPORT_PATH = "dq_checks/reports/"
MLFLOW_URI     = "mlflow/"