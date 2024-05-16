from huggingface_hub import login
from transformers import AutoTokenizer, pipeline
import torch
import streamlit as st

login(token='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')

model = "meta-llama/Llama-2-7b-chat-hf" 


tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=True)


llama_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)


st.title("Llama-2-7b Chatbot")
st.write("Enter a prompt and get a response from the Llama-2-7b model.")

# Text input for the user's prompt
user_prompt = st.text_input("Enter your prompt here:")

if user_prompt:
    # Generate sequences using the Llama-2-7b model
    sequences = llama_pipeline(
        user_prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=256,
    )

    # Display the generated text
    st.write("Chatbot:", sequences[0]['generated_text'])
