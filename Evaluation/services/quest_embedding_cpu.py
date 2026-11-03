import os
import sys
import torch
import uvicorn
sys.path.insert(0, '/root/autodl-tmp/Quest/inference')
from retrieval_model_server import create_app

torch.set_num_threads(int(os.environ.get('QUEST_EMBED_CPU_THREADS', '16')))
app = create_app('embedding', '/root/autodl-tmp/Quest/workspace/models/Qwen3-Embedding-8B', batch_size=1, max_length=1024, device='cpu')
uvicorn.run(app, host='127.0.0.1', port=18213, access_log=False)
