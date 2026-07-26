# ============================
# Import Libraries
# ============================

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

# ============================
# Model Name
# ============================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ============================
# Load Tokenizer
# ============================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ============================
# Load Model
# ============================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ============================
# Create Pipeline
# ============================

chatbot = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# ============================
# 6. Interactive Chatbot
# ============================

print("=" * 50)
print("AI Healthcare Assistant")
print("=" * 50)
print("Type exit to quit.\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    prompt = f"""
You are an AI Healthcare Assistant.

Provide clear and concise answers.
If the condition appears serious, advise consulting a healthcare professional.

Question:
{question}

Answer:
"""

    response = chatbot(
        prompt,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True
    )

    generated = response[0]["generated_text"]
    answer = generated.replace(prompt, "").strip()

    print("\nAssistant:")
    print(answer)
    
    