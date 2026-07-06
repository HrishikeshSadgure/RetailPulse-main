"""
RetailPulse - Automated Retraining DAG

Daily pipeline:
1. check_drift_task   - run Evidently drift report on latest churn data
2. retrain_task       - retrain XGBoost churn model (only if drift detected)
3. evaluate_task       - compare new model AUC against minimum threshold, promote if passing
4. log_task           - append run results to retraining_log.csv

Place this file (and pipeline_tasks.py) in the Airflow dags/ folder.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

from pipeline_tasks import (
    check_drift,
    retrain_model,
    evaluate_and_promote,
    log_pipeline_run
)


default_args = {
    "owner": "retailpulse",
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}


def _check_drift(**context):

    result = check_drift()

    context["ti"].xcom_push(key="drift_info", value=result)

    return "retrain_task" if result["retrain_needed"] else "skip_retrain_task"


def _retrain(**context):

    result = retrain_model()

    context["ti"].xcom_push(key="retrain_info", value=result)


def _evaluate(**context):

    retrain_info = context["ti"].xcom_pull(key="retrain_info", task_ids="retrain_task")

    result = evaluate_and_promote(retrain_info)

    context["ti"].xcom_push(key="eval_info", value=result)


def _log(**context):

    drift_info = context["ti"].xcom_pull(key="drift_info", task_ids="check_drift_task")

    retrain_info = context["ti"].xcom_pull(key="retrain_info", task_ids="retrain_task")

    eval_info = context["ti"].xcom_pull(key="eval_info", task_ids="evaluate_task")

    log_pipeline_run(drift_info, retrain_info, eval_info)


with DAG(
    dag_id="retailpulse_churn_retraining",
    description="Drift-triggered retraining pipeline for RetailPulse churn model",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["retailpulse", "mlops", "churn"]
) as dag:

    check_drift_task = BranchPythonOperator(
        task_id="check_drift_task",
        python_callable=_check_drift
    )

    retrain_task = PythonOperator(
        task_id="retrain_task",
        python_callable=_retrain
    )

    skip_retrain_task = EmptyOperator(
        task_id="skip_retrain_task"
    )

    evaluate_task = PythonOperator(
        task_id="evaluate_task",
        python_callable=_evaluate
    )

    log_task = PythonOperator(
        task_id="log_task",
        python_callable=_log,
        trigger_rule="none_failed_min_one_success"
    )

    check_drift_task >> [retrain_task, skip_retrain_task]
    retrain_task >> evaluate_task >> log_task
    skip_retrain_task >> log_task
