import os
import requests
import tiktoken
import numpy as np
import shutil

dataset = 'shakespeare' # ciaworld, edsheeran, haiku, trump
os_path = os.path.dirname(__file__)
base_url = 'https://huggingface.co/silaseic/nanogpt_finetuned_models/resolve/main/datasets'

def full_path(relative_pth):
    return os.path.join(os_path, relative_pth)

# choose dataset
input_file_path = full_path('input.txt')
if not os.path.exists(input_file_path):
    data_url = ''
    if dataset.lower() == 'shakespeare':
        data_url = f'{base_url}/dataset_shakespeare.txt'
    elif dataset.lower() == 'ciaworld':
        data_url = f'{base_url}/dataset_ciaworld.txt'
    elif dataset.lower() == 'edsheeran':
        data_url = f'{base_url}/dataset_edsheeran.txt'
    elif dataset.lower() == 'haiku':
        data_url = f'{base_url}/dataset_haiku.txt'
    elif dataset.lower() == 'trump':
        data_url = f'{base_url}/dataset_trump.txt'
    else:
        print(f'{dataset} does not exist, choose another one')

    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)


with open(input_file_path, 'r') as f:
    data = f.read()
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(f"dataset: {dataset}")
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens


