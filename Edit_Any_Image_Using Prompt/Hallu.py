import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer



# Function to Load model and tokenizer
def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer



# Load LLaMA and BLOOM models
llama_model_name = "meta-llama/Meta-Llama-3-8B"  # Update to correct LLaMA model if needed
bloom_model_name = "nomic-ai/gpt4all-j"  # Adjust to specific version if needed




# Ensure you have the correct access to these models
try:
    llama_model, llama_tokenizer = load_model(llama_model_name)
except Exception as e:
    print(f"Error loading LLaMA model: {e}")
    exit()

try:
    bloom_model, bloom_tokenizer = load_model(bloom_model_name)
except Exception as e:
    print(f"Error loading BLOOM model: {e}")
    exit()



# My prompts
prompts = {
    "history": [
        "what if Mughals did not Invade India",
        "Analyze the governance of British",
        "Compare the scary scenes and military strategies between world war 1 and world war 2"
        "What if it turns out the moon landings were fake?"
        "compare the egyptian civilization to Indus velly civilization"

        
    ],
    "technology": [
        "what is the main diffrence between a CPU and a GPU.",
        "Explain  higs-bosson particle",
        "Predict brain-computer interface evolution in 20 years.", 
        "Evaluate solar radiation management vs carbon capture for climate change.",
        "Is there exists evidence for extra teristrial life ?"
     
    ],
    "medicine": [
        "Explain Hemophilia ",
        "how paracetamol tablet works",
        "Explain Recombinant-DNA Insulin mechanisms in diabetes treatment"
        "Define blood plasma and can we generate it?"
        "which chemical salt is a substitute of protien ?"
        
    ]
}

# Expected facts for each prompt
expected_facts = {
    "history": [
        "Mughals", "British", "Moon", "strategyies","civilization", "fake"
    ],
    "technology": [
        "cpu", "computer", "solar", "climate", "brain"
    ],
    "medicine": [
        "DNA", "Blood", "fever", "chronic pain", "health"
    ]
}


# Generate responses
def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response



# Collect responses
llama_responses = [generate_response(llama_model, llama_tokenizer, prompt) for prompt in prompts]
bloom_responses = [generate_response(bloom_model, bloom_tokenizer, prompt) for prompt in prompts]



#  Analyze Hallucinations
def analyze_hallucinations(response, expected_facts):
    hallucinations = {'factual': 0, 'logical': 0, 'contradictory': 0}

    # Factual Hallucination Check
    for fact in expected_facts:
        if fact not in response:
            hallucinations['factual'] += 1

    # Logical Hallucination Check (simplified)
    if "but" in response or "however" in response:
        hallucinations['contradictory'] += 1

    return hallucinations



# Analyze responses for hallucinations
def analyze_responses(responses):
    results = []
    for response in responses:
        detection = analyze_hallucinations(response,expected_facts)
        results.append(detection)
    return results



llama_hallucinations = analyze_responses(llama_responses)
bloom_hallucinations = analyze_responses(bloom_responses)



# Quantitative Measurement and Analysis
def summarize_hallucinations(hallucination_results):
    summary = {
        "factual": sum(1 for result in hallucination_results if result["factual"]),
        "logical": sum(1 for result in hallucination_results if result["logical"]),
        "contradictory": sum(1 for result in hallucination_results if result["contradictory"]),
    }
    return summary



llama_summary = summarize_hallucinations(llama_hallucinations)
bloom_summary = summarize_hallucinations(bloom_hallucinations)




# Convert to DataFrame for better visualization
summary_df = pd.DataFrame({
    "Model": ["LLaMA", "CLAUDE"],
    "Factual": [llama_summary["factual"], bloom_summary["factual"]],
    "Logical": [llama_summary["logical"], bloom_summary["logical"]],
    "Contradictory": [llama_summary["contradictory"], bloom_summary["contradictory"]],
})

print(summary_df)



# Real-time detection framework
def real_time_detection(response):
    detection = analyze_hallucinations(response,expected_facts)
    if any(detection.values()):
        print("Potential hallucination detected:", detection)

# Example usage in a loop for LLaMA responses
for response in llama_responses:
    real_time_detection(response)
