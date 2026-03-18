"""
SageMaker Model Monitor Setup
================================

Detects data drift and model quality degradation in production.

CONCEPT: Why Model Monitoring?
--------------------------------
Models degrade over time because the real world changes:

  Training data: Product images from 2024 cameras, indoor lighting
  Production data (6 months later): New 2025 phone cameras, outdoor photos

This is called "data drift" — the distribution of incoming data shifts
away from what the model was trained on.

Without monitoring, accuracy silently drops from 92% to 75% and
nobody notices until customers complain.

CONCEPT: SageMaker Model Monitor
-----------------------------------
Model Monitor continuously analyzes incoming data and predictions:

  1. Baseline: Captures statistics from training data (mean pixel values,
     image sizes, class distributions, etc.)
  
  2. Monitor: Periodically compares production data to the baseline.
     If significant drift is detected, it triggers a CloudWatch alarm.

  3. Alert: CloudWatch alarm → SNS → Email/Slack notification

Types of monitoring:
  - Data Quality: Are input images changing? (size, format, pixel distribution)
  - Model Quality: Is accuracy dropping? (requires ground truth labels)
  - Bias: Are predictions unfair across subgroups?
  - Feature Attribution: Which features drive predictions?

CONCEPT: Statistical Drift Detection
---------------------------------------
Drift is detected using statistical tests:
  - KL Divergence: Measures difference between probability distributions
  - KS Test: Compares cumulative distribution functions
  - Chi-squared: Compares categorical distributions

If any metric exceeds a threshold, it's flagged as drift.

Example:
  Training: 60% of images are 1000-2000px wide
  Production: 80% of images are 500-800px wide
  → This is DATA DRIFT (images are smaller than expected)
  → Model might perform worse on smaller images
"""

import boto3
import sagemaker
from sagemaker.model_monitor import (
    DefaultModelMonitor,
    DataCaptureConfig,
    CronExpressionGenerator,
    MonitoringOutput,
)
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def enable_data_capture(
    endpoint_name: str = 'product-classifier',
    capture_bucket: str = 'ml-model-artifacts',
    sampling_percentage: int = 100
) -> DataCaptureConfig:
    """
    Enable data capture on the SageMaker endpoint.
    
    CONCEPT: Data Capture
    ----------------------
    Data capture records incoming requests and model outputs to S3.
    This creates the data that Model Monitor analyzes for drift.
    
    sampling_percentage=100 captures ALL requests. In production with
    high traffic, you might sample 10-20% to reduce storage costs.
    
    Captured data is stored as JSON Lines format in S3:
    {
        "captureData": {
            "endpointInput": {"data": "<base64 image>", "encoding": "BASE64"},
            "endpointOutput": {"data": "<json predictions>", "encoding": "JSON"}
        },
        "eventMetadata": {
            "eventId": "uuid",
            "inferenceTime": "2024-01-01T00:00:00Z"
        }
    }
    """
    return DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=sampling_percentage,
        destination_s3_uri=f's3://{capture_bucket}/data-capture/{endpoint_name}',
        capture_options=["Input", "Output"],
    )


def create_baseline(
    endpoint_name: str = 'product-classifier',
    baseline_data_uri: str = 's3://product-images-raw/train/',
    output_uri: str = 's3://ml-model-artifacts/model-monitor/baseline/',
    role: Optional[str] = None
) -> None:
    """
    Create a baseline from training data.
    
    The baseline captures the "normal" distribution of your training data.
    Model Monitor compares production data against this baseline.
    
    This is a one-time operation (or whenever you retrain with significantly
    different data).
    """
    session = sagemaker.Session()
    if role is None:
        role = sagemaker.get_execution_role()
    
    monitor = DefaultModelMonitor(
        role=role,
        instance_count=1,
        instance_type='ml.m5.large',
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
    )
    
    logger.info("Creating baseline job... (this may take 10-20 minutes)")
    
    monitor.suggest_baseline(
        baseline_dataset=baseline_data_uri,
        dataset_format={'csv': {'header': False}},
        output_s3_uri=output_uri,
    )
    
    logger.info(f"Baseline created at: {output_uri}")
    return monitor


def schedule_monitoring(
    endpoint_name: str = 'product-classifier',
    monitor: Optional[DefaultModelMonitor] = None,
    baseline_uri: str = 's3://ml-model-artifacts/model-monitor/baseline/',
    output_uri: str = 's3://ml-model-artifacts/model-monitor/reports/',
    role: Optional[str] = None
) -> None:
    """
    Schedule periodic monitoring checks.
    
    CONCEPT: Monitoring Schedule
    -----------------------------
    We run monitoring hourly. Each run:
    1. Collects data captured since the last run
    2. Computes statistics on the captured data
    3. Compares against the baseline
    4. Reports any violations (drift detected)
    5. Publishes metrics to CloudWatch
    
    If violations are found, a CloudWatch alarm triggers,
    which sends an SNS notification to the team.
    """
    session = sagemaker.Session()
    if role is None:
        role = sagemaker.get_execution_role()
    
    if monitor is None:
        monitor = DefaultModelMonitor(
            role=role,
            instance_count=1,
            instance_type='ml.m5.large',
            volume_size_in_gb=20,
            max_runtime_in_seconds=1800,
        )
    
    monitor.create_monitoring_schedule(
        monitor_schedule_name=f'{endpoint_name}-monitor-schedule',
        endpoint_input=endpoint_name,
        output_s3_uri=output_uri,
        statistics=f'{baseline_uri}/statistics.json',
        constraints=f'{baseline_uri}/constraints.json',
        schedule_cron_expression=CronExpressionGenerator.hourly(),
    )
    
    logger.info(f"Monitoring scheduled (hourly) for endpoint: {endpoint_name}")
    logger.info(f"Reports will be saved to: {output_uri}")


def check_monitoring_status(
    schedule_name: str = 'product-classifier-monitor-schedule'
) -> dict:
    """
    Check the latest monitoring execution results.
    
    Returns:
        Dictionary with monitoring status and any violations found.
    """
    sm = boto3.client('sagemaker')
    
    # Get latest execution
    executions = sm.list_monitoring_executions(
        MonitoringScheduleName=schedule_name,
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1,
    )
    
    if not executions.get('MonitoringExecutionSummaries'):
        return {'status': 'no_executions', 'message': 'No monitoring runs yet'}
    
    latest = executions['MonitoringExecutionSummaries'][0]
    
    result = {
        'status': latest['MonitoringExecutionStatus'],
        'creation_time': str(latest['CreationTime']),
        'schedule_name': schedule_name,
    }
    
    # Check for violations
    if latest['MonitoringExecutionStatus'] == 'CompletedWithViolations':
        result['violations'] = True
        result['message'] = (
            'Data drift detected! Check the monitoring report in S3. '
            'Consider retraining the model with recent data.'
        )
    elif latest['MonitoringExecutionStatus'] == 'Completed':
        result['violations'] = False
        result['message'] = 'No drift detected. Model is performing within baseline.'
    else:
        result['message'] = f"Monitoring status: {latest['MonitoringExecutionStatus']}"
    
    return result


def setup_model_monitor(endpoint_name: str = 'product-classifier') -> None:
    """
    One-click Model Monitor setup.
    
    Steps:
    1. Create baseline from training data
    2. Schedule hourly monitoring
    3. Print status and next steps
    """
    logger.info("Setting up SageMaker Model Monitor...")
    
    # Step 1: Create baseline
    monitor = create_baseline(endpoint_name=endpoint_name)
    
    # Step 2: Schedule monitoring
    schedule_monitoring(
        endpoint_name=endpoint_name,
        monitor=monitor,
    )
    
    logger.info("\nModel Monitor setup complete!")
    logger.info("Monitor will run hourly and alert on data drift.")
    logger.info("Check status: python -c \"from model_monitor import check_monitoring_status; print(check_monitoring_status())\"")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    setup_model_monitor()
