import torch
import json

from torchfly.transformers import UnifiedTokenizer

if __name__ == "__main__":
    tokenizer = UnifiedTokenizer()
    
    with open("../DialogCorpus/all_dialogs.json") as f:
        pass