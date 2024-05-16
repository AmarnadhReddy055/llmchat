import streamlit as st
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# Define model details
model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

# Download the model file from Hugging Face Hub
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

# Initialize the model
lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2,
    n_batch=2,
    n_gpu_layers=32
)

# Streamlit app interface
st.title("Llama-2 Chatbot")
st.write("Enter your prompt below and receive a response from the Llama-2 model.")

# Input prompt
prompt = st.text_area("Your prompt:")

if st.button("Generate Response"):
    # Define the prompt template
    prompt_template = f'''SYSTEM: You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible.

    USER: {prompt}

    ASSISTANT:
    '''
    
    # Generate the response
    response = lcpp_llm(
        prompt=prompt_template, 
        max_tokens=256, 
        temperature=0.5, 
        top_p=0.95,
        repeat_penalty=1.2, 
        top_k=150
    )

    # Extract and display the assistant's response
    assistant_response = response["choices"][0]["text"].split("ASSISTANT:")[1].strip()
    st.write("## Response:")
    st.write(assistant_response)
