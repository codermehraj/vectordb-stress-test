import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import sys
import json

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Define a projection matrix to transform the 384-dim embedding to 512-dim
projection_matrix = np.random.rand(384, 512)

def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
        embedding = embeddings.squeeze().numpy()
    
    # Project the 384-dim embedding to 512-dim using the projection matrix
    projected_embedding = np.dot(embedding, projection_matrix)
    
    return projected_embedding.tolist()

if __name__ == "__main__":
    text = sys.argv[1]
    embedding = generate_embedding(text)
    print(json.dumps(embedding))
