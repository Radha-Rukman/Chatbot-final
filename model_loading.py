from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

def load_models():
    model_name = "mistralai/Mistral-7B-v0.3"  # Update with your model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to('cuda')
    return tokenizer, model

def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
