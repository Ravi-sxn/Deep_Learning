from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import math




# Load pre-trained model and tokenizer
model_name = "gpt2" 
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set model to evaluation mode
model.eval()

def calculate_perplexity(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get the tokenized input IDs
    input_ids = inputs["input_ids"]

    # Generate outputs (logits)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)

    # Get the loss from the outputs
    loss = outputs.loss
    perplexity = torch.exp(loss)
    
    return perplexity.item()

# Test with a sample sentence
print("Enter Text Prompt for Perplexity")
text=str(input())

perplexity_score = calculate_perplexity(text)

print(f"Perplexity: {perplexity_score}")
