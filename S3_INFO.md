# S3 Storage — Timeweb Cloud

## Provider
**Timeweb Cloud Object Storage** — S3-compatible API

## Connection Details

| Parameter        | Value                        |
|-----------------|------------------------------|
| Endpoint        | `https://s3.twcstorage.ru`   |
| Region          | `ru-1-hot`                   |
| Protocol        | HTTPS (port 443)             |
| API             | AWS S3-compatible            |
| Bucket          | `domain-aware-sd`            |

## Credentials

Credentials are stored in `.env` (never commit to git). See `.env.example` for the template.

| Variable          | Description                       |
|------------------|-----------------------------------|
| `S3_ACCESS_KEY`  | S3 Access Key ID                  |
| `S3_SECRET_KEY`  | S3 Secret Access Key              |
| `S3_ENDPOINT`    | `https://s3.twcstorage.ru`        |
| `S3_REGION`      | `ru-1-hot`                        |
| `S3_BUCKET`      | `domain-aware-sd`                 |

> **Note**: Timeweb also provides an OpenStack Swift endpoint with a separate Swift secret key — not used in this project (we use the S3 API via boto3).

## Bucket Structure

```
domain-aware-sd/
├── models/
│   ├── TurboSparse-Mistral-Instruct/   # target model (7B)
│   └── tiny-mixtral/                   # compatible draft model
├── data/
│   └── synthetic/
│       └── v1/                         # generated data (one .jsonl per cluster)
│           ├── aeslc_10templates.jsonl
│           ├── ag_news_subset_10templates.jsonl
│           └── ...  (66 files total)
└── checkpoints/
    └── drafters/                        # fine-tuned domain-specific draft models
        └── <cluster_name>/
```

## Using boto3

```python
import boto3, os
from dotenv import load_dotenv

load_dotenv(".env")

s3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("S3_ENDPOINT"),
    aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
    region_name=os.getenv("S3_REGION"),
)

# List objects
response = s3.list_objects_v2(Bucket="domain-aware-sd", Prefix="models/")

# Upload a file
s3.upload_file("local/path/file.txt", "domain-aware-sd", "remote/key/file.txt")

# Download a file
s3.download_file("domain-aware-sd", "remote/key/file.txt", "local/path/file.txt")

# Generate a presigned URL (valid for 1 hour)
url = s3.generate_presigned_url(
    "get_object",
    Params={"Bucket": "domain-aware-sd", "Key": "models/tiny-mixtral/config.json"},
    ExpiresIn=3600,
)
```

## Using AWS CLI

```bash
# Configure profile (one-time)
aws configure --profile timeweb
# Access Key ID: <S3_ACCESS_KEY>
# Secret Access Key: <S3_SECRET_KEY>
# Region: ru-1-hot

# Then use with --endpoint-url
aws s3 ls s3://domain-aware-sd/ --endpoint-url https://s3.twcstorage.ru --profile timeweb
aws s3 cp local_file s3://domain-aware-sd/path/ --endpoint-url https://s3.twcstorage.ru --profile timeweb
aws s3 sync ./models/tiny-mixtral s3://domain-aware-sd/models/tiny-mixtral --endpoint-url https://s3.twcstorage.ru --profile timeweb
```

## Creating the Bucket

If the bucket doesn't exist yet, create it before uploading:

```python
s3.create_bucket(Bucket="domain-aware-sd")
```

Or via CLI:
```bash
aws s3 mb s3://domain-aware-sd --endpoint-url https://s3.twcstorage.ru --profile timeweb
```

## Cost / Limits

Timeweb Cloud charges per GB stored and per GB transferred. Keep large checkpoints in `/checkpoints/` and use lifecycle rules or manual cleanup to avoid accumulating stale data.
