"""
AWS Helper Utilities
=====================

Reusable functions for interacting with AWS services.
Centralizes AWS operations so they're consistent across the codebase.

CONCEPT: Utility Modules
--------------------------
Instead of scattering boto3 calls throughout the codebase, we centralize
them here. Benefits:
  - One place to update if AWS APIs change
  - Consistent error handling
  - Easy to mock for testing
  - DRY (Don't Repeat Yourself)
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch, PyTorchModel
import json
import os
import logging
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def get_sagemaker_session() -> sagemaker.Session:
    """Create a SageMaker session with default settings."""
    return sagemaker.Session()


def get_execution_role() -> str:
    """
    Get the SageMaker execution role ARN.
    
    CONCEPT: IAM Roles
    -------------------
    SageMaker needs an IAM role to:
    - Read training data from S3
    - Write model artifacts to S3
    - Create CloudWatch logs
    - Access ECR for Docker images
    
    The role is like a "permission badge" — it defines what SageMaker can do.
    We use sagemaker.get_execution_role() in notebooks, but for scripts
    we read from environment variables.
    """
    role = os.environ.get('SAGEMAKER_ROLE_ARN')
    if role:
        return role
    
    try:
        return sagemaker.get_execution_role()
    except ValueError:
        logger.warning(
            "Could not auto-detect SageMaker role. "
            "Set SAGEMAKER_ROLE_ARN environment variable."
        )
        raise


def launch_training_job(
    config_path: str = 'configs/training_config.yaml',
    wait: bool = False
) -> str:
    """
    Launch a SageMaker training job.
    
    CONCEPT: SageMaker Training Jobs
    ----------------------------------
    A training job:
    1. Provisions the specified EC2 instance (e.g., ml.g4dn.xlarge)
    2. Downloads your training script from S3
    3. Downloads training data from S3
    4. Runs the training script with specified hyperparameters
    5. Uploads model artifacts to S3
    6. Terminates the instance (you only pay for training time!)
    
    CONCEPT: Spot Instances
    ------------------------
    Spot instances use unused AWS capacity at up to 70% discount.
    The catch: AWS can terminate them with 2-minute warning.
    SageMaker handles checkpointing — if interrupted, it resumes
    from the last checkpoint. For training, this is almost always worth it.
    
    Args:
        config_path: Path to training configuration YAML
        wait: If True, block until training completes
    
    Returns:
        Training job name
    """
    import yaml
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    session = get_sagemaker_session()
    role = get_execution_role()
    
    # Configure the training job
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='src/training/',
        role=role,
        framework_version='2.0.0',
        py_version='py310',
        instance_count=1,
        instance_type=config['sagemaker']['training_instance'],
        use_spot_instances=config['sagemaker']['spot_training'],
        max_wait=config['sagemaker']['max_runtime_seconds'] * 2 if config['sagemaker']['spot_training'] else None,
        max_run=config['sagemaker']['max_runtime_seconds'],
        volume_size=config['sagemaker']['volume_size_gb'],
        hyperparameters={
            'epochs': config['training']['epochs'],
            'batch-size': config['training']['batch_size'],
            'learning-rate': config['training']['learning_rate'],
            'weight-decay': config['training']['weight_decay'],
            'dropout-rate': config['model']['dropout_rate'],
            'freeze-layers': config['model']['freeze_layers'],
            'num-classes': config['model']['num_classes'],
            'patience': config['training']['patience'],
        },
        sagemaker_session=session,
    )
    
    # Start training
    s3_config = config['s3']
    estimator.fit(
        inputs={
            'train': f"s3://{s3_config['raw_bucket']}/{s3_config['train_prefix']}",
            'validation': f"s3://{s3_config['raw_bucket']}/{s3_config['val_prefix']}",
        },
        wait=wait,
    )
    
    job_name = estimator.latest_training_job.name
    logger.info(f"Training job launched: {job_name}")
    
    if wait:
        logger.info(f"Model artifacts: {estimator.model_data}")
    
    return job_name


def deploy_endpoint(
    model_data: Optional[str] = None,
    endpoint_name: str = 'product-classifier',
    instance_type: str = 'ml.m5.large',
    initial_instance_count: int = 1
) -> str:
    """
    Deploy a trained model to a SageMaker endpoint.
    
    CONCEPT: SageMaker Endpoints
    ------------------------------
    An endpoint is a hosted model that accepts real-time inference requests.
    
    How it works:
    1. SageMaker provisions EC2 instance(s)
    2. Downloads model artifacts from S3
    3. Loads model into memory (runs model_fn from inference.py)
    4. Accepts HTTPS requests (runs input_fn → predict_fn → output_fn)
    5. Auto-scales based on traffic (if configured)
    
    Cost: You pay for the instance(s) running 24/7, even without requests.
    That's why auto-scaling to 0 during idle periods is important.
    
    Args:
        model_data: S3 URI of model.tar.gz (from training job)
        endpoint_name: Name for the endpoint
        instance_type: EC2 instance type for inference
        initial_instance_count: Starting number of instances
    
    Returns:
        Endpoint name
    """
    role = get_execution_role()
    
    model = PyTorchModel(
        model_data=model_data,
        role=role,
        entry_point='inference.py',
        source_dir='src/inference/',
        framework_version='2.0.0',
        py_version='py310',
    )
    
    predictor = model.deploy(
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )
    
    logger.info(f"Endpoint deployed: {endpoint_name}")
    return endpoint_name


def setup_autoscaling(
    endpoint_name: str = 'product-classifier',
    min_capacity: int = 1,
    max_capacity: int = 5,
    target_invocations: float = 75.0
) -> None:
    """
    Configure auto-scaling for SageMaker endpoint.
    
    CONCEPT: Auto-Scaling
    ----------------------
    Auto-scaling automatically adjusts the number of instances based on traffic.
    
    Our policy: Target 75 invocations per instance.
    - < 75 invocations/instance: Scale down (save money)
    - > 75 invocations/instance: Scale up (maintain latency)
    
    Cooldown periods prevent thrashing:
    - Scale-out cooldown (60s): Wait 1 min between adding instances
    - Scale-in cooldown (300s): Wait 5 min between removing instances
    
    The asymmetry is intentional:
    - Scale OUT quickly (users are waiting!)
    - Scale IN slowly (traffic might come back)
    """
    client = boto3.client('application-autoscaling')
    
    resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"
    
    # Register the endpoint as a scalable target
    client.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MinCapacity=min_capacity,
        MaxCapacity=max_capacity,
    )
    
    # Create target tracking scaling policy
    client.put_scaling_policy(
        PolicyName=f'{endpoint_name}-scaling-policy',
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'TargetValue': target_invocations,
            'PredefinedMetricSpecification': {
                'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
            },
            'ScaleInCooldown': 300,   # 5 min before removing instances
            'ScaleOutCooldown': 60,   # 1 min before adding instances
        }
    )
    
    logger.info(
        f"Auto-scaling enabled: {min_capacity}-{max_capacity} instances, "
        f"target {target_invocations} invocations/instance"
    )


def create_s3_buckets(environment: str = 'dev') -> Dict[str, str]:
    """
    Create required S3 buckets with proper configuration.
    
    CONCEPT: S3 Bucket Design
    ---------------------------
    We use three buckets to separate concerns:
    1. Raw: Original uploaded images (versioned, lifecycle to Glacier)
    2. Processed: Resized/normalized images (temporary, auto-deleted)
    3. Artifacts: Model weights, training logs (versioned, permanent)
    """
    s3 = boto3.client('s3')
    account_id = boto3.client('sts').get_caller_identity()['Account']
    region = boto3.session.Session().region_name
    
    buckets = {
        'raw': f"product-images-raw-{environment}-{account_id}",
        'processed': f"product-images-processed-{environment}-{account_id}",
        'artifacts': f"ml-model-artifacts-{environment}-{account_id}",
    }
    
    for purpose, bucket_name in buckets.items():
        try:
            if region == 'us-east-1':
                s3.create_bucket(Bucket=bucket_name)
            else:
                s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )
            
            # Enable versioning on raw and artifacts
            if purpose in ('raw', 'artifacts'):
                s3.put_bucket_versioning(
                    Bucket=bucket_name,
                    VersioningConfiguration={'Status': 'Enabled'}
                )
            
            # Block public access
            s3.put_public_access_block(
                Bucket=bucket_name,
                PublicAccessBlockConfiguration={
                    'BlockPublicAcls': True,
                    'IgnorePublicAcls': True,
                    'BlockPublicPolicy': True,
                    'RestrictPublicBuckets': True,
                }
            )
            
            logger.info(f"Created bucket: {bucket_name} ({purpose})")
            
        except s3.exceptions.BucketAlreadyOwnedByYou:
            logger.info(f"Bucket already exists: {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to create bucket {bucket_name}: {e}")
            raise
    
    return buckets


def create_dynamodb_table(
    table_name: str = 'product-predictions',
    environment: str = 'dev'
) -> str:
    """
    Create the DynamoDB predictions table.
    
    Returns the full table name.
    """
    dynamodb = boto3.client('dynamodb')
    full_name = f"{table_name}-{environment}"
    
    try:
        dynamodb.create_table(
            TableName=full_name,
            KeySchema=[
                {'AttributeName': 'product_id', 'KeyType': 'HASH'},
                {'AttributeName': 'timestamp', 'KeyType': 'RANGE'},
            ],
            AttributeDefinitions=[
                {'AttributeName': 'product_id', 'AttributeType': 'S'},
                {'AttributeName': 'timestamp', 'AttributeType': 'S'},
                {'AttributeName': 'category', 'AttributeType': 'S'},
            ],
            GlobalSecondaryIndexes=[{
                'IndexName': 'category-index',
                'KeySchema': [{'AttributeName': 'category', 'KeyType': 'HASH'}],
                'Projection': {'ProjectionType': 'ALL'},
            }],
            BillingMode='PAY_PER_REQUEST',
            StreamSpecification={
                'StreamEnabled': True,
                'StreamViewType': 'NEW_AND_OLD_IMAGES',
            },
        )
        
        # Enable TTL
        dynamodb.update_time_to_live(
            TableName=full_name,
            TimeToLiveSpecification={
                'Enabled': True,
                'AttributeName': 'ttl',
            }
        )
        
        logger.info(f"Created DynamoDB table: {full_name}")
        
    except dynamodb.exceptions.ResourceInUseException:
        logger.info(f"Table already exists: {full_name}")
    
    return full_name
