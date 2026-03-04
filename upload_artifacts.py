#!/usr/bin/env python3
"""
Upload artifacts to HuggingFace
"""

import os
from huggingface_hub import HfApi

def main():
    api = HfApi(token=os.getenv('HF_TOKEN'))
    repo_id = 'P2SAMAPA/etf-entropy-dataset'
    
    artifact_path = 'artifacts'
    
    for file in os.listdir(artifact_path):
        if file.endswith(('.pkl', '.json', '.npy')):
            print(f'Uploading {file}...')
            api.upload_file(
                path_or_fileobj=f'{artifact_path}/{file}',
                path_in_repo=f'models/{file}',
                repo_id=repo_id,
                repo_type='dataset'
            )
    
    print('All artifacts uploaded to HF!')

if __name__ == "__main__":
    main()
