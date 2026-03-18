"""
CloudWatch Monitoring & Alerting Setup
========================================

Sets up comprehensive monitoring for the ML pipeline:
- Dashboard with key metrics
- Alarms for anomaly detection
- SNS notifications for on-call engineers

CONCEPT: Why Monitoring Matters for ML Systems
------------------------------------------------
ML systems fail SILENTLY. Unlike a web server that crashes with a 500 error,
an ML model can degrade gradually:
  - Model drift: Input data changes over time (new product types, seasonal trends)
  - Data drift: Image quality changes (new phone cameras, different lighting)
  - Latency creep: Model slows down as data changes
  - Silent failures: Model always predicts "electronics" but nobody notices

Monitoring catches these issues BEFORE they impact customers.

CONCEPT: The Four Golden Signals (from Google SRE)
----------------------------------------------------
1. Latency: How long do requests take? (p50, p90, p99)
2. Traffic: How many requests are coming? (requests/second)
3. Errors: What percentage of requests fail?
4. Saturation: How "full" is the system? (CPU, memory, queue depth)

We monitor all four for both the Lambda function and SageMaker endpoint.

CONCEPT: CloudWatch
---------------------
CloudWatch is AWS's monitoring service. It collects:
  - Metrics: Numerical data points over time (latency, error count, CPU %)
  - Logs: Text logs from Lambda, SageMaker, API Gateway
  - Alarms: Rules that trigger notifications when metrics cross thresholds
  - Dashboards: Visual displays of metrics

Key terms:
  - Namespace: Category of metrics (e.g., AWS/Lambda, AWS/SageMaker)
  - Dimension: Filters metrics (e.g., FunctionName=classify-product)
  - Period: Time window for aggregation (e.g., 300 seconds = 5 minutes)
  - Statistic: How to aggregate (Average, Sum, Maximum, p99, etc.)
"""

import boto3
import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

cloudwatch = boto3.client('cloudwatch')
sns = boto3.client('sns')


def create_sns_topic(topic_name: str = 'ml-pipeline-alerts') -> str:
    """
    Create SNS topic for alert notifications.
    
    CONCEPT: SNS (Simple Notification Service)
    --------------------------------------------
    SNS is a pub/sub messaging service:
    - Topic: A channel for messages (like a mailing list)
    - Subscription: Who receives messages (email, SMS, Lambda, Slack webhook)
    - Message: The alert content
    
    Flow: CloudWatch Alarm triggers → SNS Topic → Your Email/Slack/PagerDuty
    """
    response = sns.create_topic(Name=topic_name)
    topic_arn = response['TopicArn']
    
    logger.info(f"Created SNS topic: {topic_arn}")
    logger.info("Subscribe your email: aws sns subscribe --topic-arn <ARN> --protocol email --notification-endpoint your@email.com")
    
    return topic_arn


def create_dashboard(
    dashboard_name: str = 'ProductClassifierDashboard',
    region: str = 'us-east-1'
) -> None:
    """
    Create a CloudWatch dashboard with all key metrics.
    
    CONCEPT: Dashboard Design
    --------------------------
    A good ML dashboard answers these questions at a glance:
    1. Is the system healthy RIGHT NOW? (latency, errors)
    2. How much traffic are we handling? (invocations)
    3. Is the model performing well? (prediction distribution)
    4. Are we within budget? (costs per service)
    """
    dashboard_body = {
        "widgets": [
            # ================================================================
            # Row 1: SageMaker Endpoint Health
            # ================================================================
            {
                "type": "metric",
                "x": 0, "y": 0, "width": 12, "height": 6,
                "properties": {
                    "title": "SageMaker Endpoint - Latency",
                    "metrics": [
                        # CONCEPT: Percentile Metrics
                        # p50 = median (50% of requests are faster)
                        # p90 = 90th percentile (90% are faster)
                        # p99 = 99th percentile (only 1% are slower)
                        # p99 is what matters for SLAs — it shows worst-case experience
                        ["AWS/SageMaker", "ModelLatency", "EndpointName", "product-classifier",
                         "VariantName", "AllTraffic", {"stat": "p50", "label": "p50 Latency"}],
                        ["...", {"stat": "p90", "label": "p90 Latency"}],
                        ["...", {"stat": "p99", "label": "p99 Latency"}],
                    ],
                    "period": 300,
                    "region": region,
                    "yAxis": {"left": {"label": "Milliseconds", "showUnits": False}},
                    "view": "timeSeries",
                    "stacked": False
                }
            },
            {
                "type": "metric",
                "x": 12, "y": 0, "width": 12, "height": 6,
                "properties": {
                    "title": "SageMaker Endpoint - Invocations & Errors",
                    "metrics": [
                        ["AWS/SageMaker", "Invocations", "EndpointName", "product-classifier",
                         "VariantName", "AllTraffic", {"stat": "Sum", "label": "Total Invocations"}],
                        [".", "InvocationErrors", ".", ".", ".", ".",
                         {"stat": "Sum", "label": "Errors", "color": "#d62728"}],
                    ],
                    "period": 300,
                    "region": region,
                    "view": "timeSeries"
                }
            },
            
            # ================================================================
            # Row 2: Lambda Function Metrics
            # ================================================================
            {
                "type": "metric",
                "x": 0, "y": 6, "width": 8, "height": 6,
                "properties": {
                    "title": "Lambda - Duration",
                    "metrics": [
                        ["AWS/Lambda", "Duration", "FunctionName", "product-classifier-function",
                         {"stat": "Average", "label": "Avg Duration"}],
                        ["...", {"stat": "Maximum", "label": "Max Duration"}],
                    ],
                    "period": 300,
                    "region": region,
                    "yAxis": {"left": {"label": "ms"}},
                    "view": "timeSeries"
                }
            },
            {
                "type": "metric",
                "x": 8, "y": 6, "width": 8, "height": 6,
                "properties": {
                    "title": "Lambda - Errors & Throttles",
                    "metrics": [
                        ["AWS/Lambda", "Errors", "FunctionName", "product-classifier-function",
                         {"stat": "Sum", "label": "Errors", "color": "#d62728"}],
                        [".", "Throttles", ".", ".",
                         {"stat": "Sum", "label": "Throttles", "color": "#ff7f0e"}],
                    ],
                    "period": 300,
                    "region": region,
                    "view": "timeSeries"
                }
            },
            {
                "type": "metric",
                "x": 16, "y": 6, "width": 8, "height": 6,
                "properties": {
                    "title": "Lambda - Concurrent Executions",
                    "metrics": [
                        ["AWS/Lambda", "ConcurrentExecutions", "FunctionName", "product-classifier-function",
                         {"stat": "Maximum", "label": "Peak Concurrency"}],
                    ],
                    "period": 60,
                    "region": region,
                    "view": "timeSeries"
                }
            },
            
            # ================================================================
            # Row 3: API Gateway & DynamoDB
            # ================================================================
            {
                "type": "metric",
                "x": 0, "y": 12, "width": 12, "height": 6,
                "properties": {
                    "title": "API Gateway - Requests & Latency",
                    "metrics": [
                        ["AWS/ApiGateway", "Count", "ApiName", "ProductClassifierAPI",
                         {"stat": "Sum", "label": "Total Requests"}],
                        [".", "Latency", ".", ".",
                         {"stat": "Average", "label": "Avg Latency", "yAxis": "right"}],
                        [".", "4XXError", ".", ".",
                         {"stat": "Sum", "label": "4XX Errors", "color": "#ff7f0e"}],
                        [".", "5XXError", ".", ".",
                         {"stat": "Sum", "label": "5XX Errors", "color": "#d62728"}],
                    ],
                    "period": 300,
                    "region": region,
                    "view": "timeSeries"
                }
            },
            {
                "type": "metric",
                "x": 12, "y": 12, "width": 12, "height": 6,
                "properties": {
                    "title": "DynamoDB - Read/Write Capacity",
                    "metrics": [
                        ["AWS/DynamoDB", "ConsumedWriteCapacityUnits", "TableName", "product-predictions",
                         {"stat": "Sum", "label": "Write Units"}],
                        [".", "ConsumedReadCapacityUnits", ".", ".",
                         {"stat": "Sum", "label": "Read Units"}],
                    ],
                    "period": 300,
                    "region": region,
                    "view": "timeSeries"
                }
            },
        ]
    }
    
    cloudwatch.put_dashboard(
        DashboardName=dashboard_name,
        DashboardBody=json.dumps(dashboard_body)
    )
    
    logger.info(f"Dashboard created: {dashboard_name}")
    logger.info(f"View at: https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}#dashboards:name={dashboard_name}")


def create_alarms(alert_topic_arn: str) -> None:
    """
    Create CloudWatch alarms for critical metrics.
    
    CONCEPT: Alarm Design
    ----------------------
    Good alarms are:
    - Actionable: Someone can DO something when it fires
    - Specific: Points to the problem area
    - Not noisy: Doesn't fire for minor fluctuations
    
    We use EvaluationPeriods > 1 to avoid false alarms from momentary spikes.
    Example: Threshold of 1000ms for 3 consecutive 5-min periods
             = latency must be high for 15+ minutes before alerting
    """
    
    alarms = [
        # Alarm 1: High inference latency
        {
            'AlarmName': 'ProductClassifier-HighLatency',
            'AlarmDescription': 'SageMaker endpoint p99 latency exceeds 1 second for 15+ minutes',
            'MetricName': 'ModelLatency',
            'Namespace': 'AWS/SageMaker',
            'Dimensions': [
                {'Name': 'EndpointName', 'Value': 'product-classifier'},
                {'Name': 'VariantName', 'Value': 'AllTraffic'},
            ],
            'Statistic': 'Average',
            'Period': 300,          # 5-minute windows
            'EvaluationPeriods': 3, # Must be high for 3 consecutive periods (15 min)
            'Threshold': 1000.0,    # 1 second
            'ComparisonOperator': 'GreaterThanThreshold',
            'TreatMissingData': 'notBreaching',  # No data = no traffic = OK
        },
        
        # Alarm 2: High error rate on Lambda
        {
            'AlarmName': 'ProductClassifier-HighErrorRate',
            'AlarmDescription': 'Lambda function error count exceeds 10 in 5 minutes',
            'MetricName': 'Errors',
            'Namespace': 'AWS/Lambda',
            'Dimensions': [
                {'Name': 'FunctionName', 'Value': 'product-classifier-function'},
            ],
            'Statistic': 'Sum',
            'Period': 300,
            'EvaluationPeriods': 1,  # Alert immediately — errors are serious
            'Threshold': 10.0,
            'ComparisonOperator': 'GreaterThanThreshold',
            'TreatMissingData': 'notBreaching',
        },
        
        # Alarm 3: Lambda throttling (hitting concurrency limits)
        {
            'AlarmName': 'ProductClassifier-LambdaThrottles',
            'AlarmDescription': 'Lambda function being throttled — may need higher concurrency limit',
            'MetricName': 'Throttles',
            'Namespace': 'AWS/Lambda',
            'Dimensions': [
                {'Name': 'FunctionName', 'Value': 'product-classifier-function'},
            ],
            'Statistic': 'Sum',
            'Period': 300,
            'EvaluationPeriods': 2,
            'Threshold': 5.0,
            'ComparisonOperator': 'GreaterThanThreshold',
            'TreatMissingData': 'notBreaching',
        },
        
        # Alarm 4: API Gateway 5XX errors
        {
            'AlarmName': 'ProductClassifier-API5xxErrors',
            'AlarmDescription': 'API Gateway returning 5XX errors — server-side failure',
            'MetricName': '5XXError',
            'Namespace': 'AWS/ApiGateway',
            'Dimensions': [
                {'Name': 'ApiName', 'Value': 'ProductClassifierAPI'},
            ],
            'Statistic': 'Sum',
            'Period': 300,
            'EvaluationPeriods': 1,
            'Threshold': 5.0,
            'ComparisonOperator': 'GreaterThanThreshold',
            'TreatMissingData': 'notBreaching',
        },
        
        # Alarm 5: SageMaker endpoint invocation errors
        {
            'AlarmName': 'ProductClassifier-EndpointErrors',
            'AlarmDescription': 'SageMaker endpoint returning errors — model may be broken',
            'MetricName': 'InvocationErrors',
            'Namespace': 'AWS/SageMaker',
            'Dimensions': [
                {'Name': 'EndpointName', 'Value': 'product-classifier'},
                {'Name': 'VariantName', 'Value': 'AllTraffic'},
            ],
            'Statistic': 'Sum',
            'Period': 300,
            'EvaluationPeriods': 1,
            'Threshold': 3.0,
            'ComparisonOperator': 'GreaterThanThreshold',
            'TreatMissingData': 'notBreaching',
        },
    ]
    
    for alarm_config in alarms:
        cloudwatch.put_metric_alarm(
            **alarm_config,
            AlarmActions=[alert_topic_arn],
            OKActions=[alert_topic_arn],  # Notify when alarm clears too
        )
        logger.info(f"Created alarm: {alarm_config['AlarmName']}")
    
    logger.info(f"All {len(alarms)} alarms created with notifications to {alert_topic_arn}")


def setup_monitoring():
    """One-click monitoring setup."""
    logger.info("Setting up monitoring infrastructure...")
    
    # Create SNS topic
    topic_arn = create_sns_topic()
    
    # Create dashboard
    create_dashboard()
    
    # Create alarms
    create_alarms(topic_arn)
    
    logger.info("\nMonitoring setup complete!")
    logger.info("Next steps:")
    logger.info("  1. Subscribe to SNS topic for email alerts")
    logger.info("  2. Check dashboard in CloudWatch console")
    logger.info("  3. Test alarms with: aws cloudwatch set-alarm-state --alarm-name <name> --state-value ALARM --state-reason 'testing'")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    setup_monitoring()
