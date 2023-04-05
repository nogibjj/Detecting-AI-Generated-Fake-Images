import boto3
import os
from dotenv import load_dotenv

load_dotenv()
session = boto3.Session(
    aws_access_key_id = os.getenv('AWS_Access_Key_ID'),
    aws_secret_access_key = os.getenv('AWS_Secret_Access_Key'),
    region_name='us-west-2'
)
s3 = session.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)
