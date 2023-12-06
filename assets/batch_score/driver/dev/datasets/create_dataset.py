# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# https://huggingface.co/datasets/cnn_dailymail
from datasets import load_dataset
import pandas as pd
import os
import mltable

dataname = 'cnndailymail1k'
os.mkdir(dataname)

data = load_dataset('cnn_dailymail', "3.0.0", split='test', streaming=True).take(1000)
df = pd.DataFrame(data)
df.rename(columns={'article': 'input'}, inplace=True)
df = df[['input']]


df.to_json(f'{dataname}/{dataname}.jsonl', orient='records', lines=True)


m = mltable.from_json_lines_files([
     {'file':f'{dataname}/{dataname}.jsonl'}
    ])
m.save(dataname)