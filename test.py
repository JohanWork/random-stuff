from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModel

# Load dataset from the hub
dataset = load_dataset("philschmid/finanical-rag-embedding-dataset", split="train")
 
# rename columns
dataset = dataset.rename_column("question", "anchor")
dataset = dataset.rename_column("context", "positive")

model_name = "answerdotai/ModernBERT-base"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def get_embeddings(text):
    # Tokenize and prepare input
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    
    # Move inputs to GPU if available
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use CLS token embedding (first token) and move back to CPU for numpy conversion
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings[0]

# Create embeddings for the anchor column
dataset = dataset.map(
    lambda x: {"embeddings": get_embeddings(x["anchor"])},
)

# You can verify the new column exists and see the first few examples
print(dataset.column_names)  # Should show 'embeddings' in the list
print(dataset['embeddings'][0])  # Print first embedding vector


