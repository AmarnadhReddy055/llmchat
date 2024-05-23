import streamlit as st
from huggingface_hub import login
from transformers import AutoTokenizer, pipeline
import torch
import accelerate
# Function to generate text using Llama model
def generate_text(prompt, llama_pipeline, tokenizer):
    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=256,
    )
    generated_text = sequences[0]['generated_text']
    return generated_text

# Login to Hugging Face
login(token='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')

# Load the model and tokenizer
model = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)
llama_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,  # Pass tokenizer to pipeline
    torch_dtype=torch.float16,
    device_map="auto",
)

# Streamlit app initialization
st.title("Text Generation with Llama Model")

# Input prompt from the user
prompt = st.text_area("Enter your prompt here:")

# Generate text when the user clicks the button
if st.button("Generate Text"):
    if not prompt:
        st.error("Prompt is required")
    else:
        generated_text = generate_text(prompt, llama_pipeline, tokenizer)
        st.write("Generated Text:")
        st.write(generated_text)
