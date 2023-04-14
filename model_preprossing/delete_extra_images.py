import numpy as np
import boto3

import pandas as pd

session = boto3.Session(
    aws_access_key_id="AKIAV3KKLC57NGTGPB7K",
    aws_secret_access_key="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",
    region_name='us-east-1'
)

s3 = session.client('s3')

paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket='dalle2images', Prefix='fake/')
pages_real = paginator.paginate(Bucket='dalle2images', Prefix='real/')

fake_images = list()
real_images = list()

for page in pages:
    for image in page["Contents"]:
        image_name = image["Key"]
        image_name = image_name.split("/")[1]
        fake_images.append(image_name)

for page in pages_real:
    for image in page["Contents"]:
        image_name = image["Key"]
        image_name = image_name.split("/")[1]
        real_images.append(image_name)

fake_images = fake_images[1:]
real_images = real_images[1:]

extra_real = [i for i in real_images if i not in fake_images]

for i in extra_real:
    s3.delete_object(Bucket='dalle2images', Key="real/"+i)

print(fake_images[:5], real_images[:5], extra_real[:5])
print(len(fake_images), len(real_images), len(extra_real))

real_after = list()
pages_real_after = paginator.paginate(Bucket='dalle2images', Prefix='real/')

for page in pages_real_after:
    for image in page["Contents"]:
        image_name = image["Key"]
        image_name = image_name.split("/")[1]
        real_after.append(image_name)

real_after = real_after[1:]
print(len(real_after), real_after[:5])