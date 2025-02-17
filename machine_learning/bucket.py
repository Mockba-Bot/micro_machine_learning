import os
import sys
from dotenv import load_dotenv
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

# Load environment variables from the .env file
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env.micro.machine.learning'))
load_dotenv(dotenv_path=dotenv_path)


# Configuration for DigitalOcean Spaces
OBJECT_STORAGE_URL = os.getenv("OBJECT_STORAGE_URL")  # Your DigitalOcean endpoint URL
ACCESS_KEY = os.getenv("ACCESS_KEY")  # Replace with your DigitalOcean Spaces access key
SECRET_KEY = os.getenv("SECRET_KEY")  # Replace with your DigitalOcean Spaces secret key
BUCKET_NAME = os.getenv("BUCKET_NAME")  # Your bucket name
REGION_NAME = os.getenv("REGION_NAME")  # Your region name

# Initialize the S3 client with explicit region configuration
s3_client = boto3.client(
    's3', 
    endpoint_url=OBJECT_STORAGE_URL,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name=REGION_NAME  # Ensure region is set
)

def download_model(bucket_name, key, local_path):
    """Download a file from DigitalOcean Spaces."""
    try:
        s3_client.download_file(bucket_name, key, local_path)
        print(f"Model downloaded to {local_path}")
        return True
    except Exception as e:
        print(f"Model not found it needs to be trained")
        return False

def upload_model(bucket_name, key, local_path):
    """Upload a file to DigitalOcean Spaces."""
    try:
        s3_client.upload_file(local_path, bucket_name, key)
        print(f"Model uploaded to {bucket_name}/{key}")
    except Exception as e:
        print(f"Error uploading model: {e}")