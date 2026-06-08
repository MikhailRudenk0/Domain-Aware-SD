# Public Read Access — S3 Bucket

The `domain-aware-sd` bucket on Timeweb Cloud Object Storage is publicly readable. No credentials required.

| Parameter | Value |
|-----------|-------|
| Endpoint  | `https://s3.twcstorage.ru` |
| Region    | `ru-1-hot` |
| Bucket    | `domain-aware-sd` |

## Download a file (AWS CLI)

```bash
aws s3 cp s3://domain-aware-sd/<key> ./ \
  --endpoint-url https://s3.twcstorage.ru \
  --no-sign-request
```

## List bucket contents (AWS CLI)

```bash
aws s3 ls s3://domain-aware-sd/ \
  --endpoint-url https://s3.twcstorage.ru \
  --no-sign-request
```

## Download with boto3 (Python)

```python
import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.twcstorage.ru",
    region_name="ru-1-hot",
    config=Config(signature_version=UNSIGNED),
)

# Download a file
s3.download_file("domain-aware-sd", "<key>", "local_filename")
```

## Direct HTTPS URL

Files can also be fetched directly:

```
https://s3.twcstorage.ru/domain-aware-sd/<key>
```

## Bucket structure

```
domain-aware-sd/
├── models/
│   ├── TurboSparse-Mistral-Instruct/   # target model (7B)
│   └── tiny-mixtral/                   # draft model
├── data/
│   └── synthetic/v1/                   # generated data (66 .jsonl files)
└── checkpoints/
    └── drafters/                        # fine-tuned domain-specific drafters
```
