import os
import requests
import tiktoken
import numpy as np
import shutil

dataset = 'shakespeare' # ciaworld, edsheeran, haiku, trump
os_path = os.path.dirname(__file__)


def full_path(relative_pth):
    return os.path.join(os_path, relative_pth)

# choose dataset
input_file_path = full_path('input.txt')
if not os.path.exists(input_file_path):
    output_pth = full_path('input.txt')
    if dataset.lower() == 'shakespeare':
        shutil.copyfile(full_path('../../datasets/dataset_shakespeare.txt'), output_pth)
    elif dataset.lower() == 'ciaworld':
        shutil.copyfile(full_path('../../datasets/dataset_ciaworld.txt'), output_pth)
    elif dataset.lower() == 'edsheeran':
        shutil.copyfile(full_path('../../datasets/dataset_edsheeran.txt'), output_pth)
    elif dataset.lower() == 'haiku':
        shutil.copyfile(full_path('../../datasets/dataset_haiku.txt'), output_pth)
    elif dataset.lower() == 'trump':
        shutil.copyfile(full_path('../../datasets/dataset_trump.txt'), output_pth)
    else:
        print(f'{dataset} does not exist, choose another one')


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


