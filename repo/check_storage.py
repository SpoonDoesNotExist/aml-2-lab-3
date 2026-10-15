"""Снимок объектов MinIO и локальных артефактов для сравнения запусков."""
import hashlib
import json
from pathlib import Path

import boto3
import yaml
from airflow.models import Variable

s3 = boto3.client(
    's3', endpoint_url=Variable.get('minio_endpoint'),
    aws_access_key_id=Variable.get('minio_access_key'),
    aws_secret_access_key=Variable.get('minio_secret_key'), region_name='us-east-1',
)
objects = s3.list_objects_v2(Bucket='titanic-dvc').get('Contents', [])
result = {'objects': [], 'artifacts': []}
for item in sorted(objects, key=lambda value: value['Key']):
    result['objects'].append({
        'key': item['Key'], 'size': item['Size'], 'etag': item['ETag'],
        'last_modified': item['LastModified'].isoformat(),
    })
for name in ['data_train.csv', 'data_test.csv', 'model.pkl']:
    path = Path(name)
    md5 = yaml.safe_load(Path(name + '.dvc').read_text())['outs'][0]['md5']
    key = f'files/md5/{md5[:2]}/{md5[2:]}'
    remote = s3.get_object(Bucket='titanic-dvc', Key=key)['Body'].read()
    assert remote == path.read_bytes(), name
    result['artifacts'].append({
        'name': name, 'md5': md5, 'key': key,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'remote_bytes_match': True,
    })
print(json.dumps(result, indent=2))
