from flask import Flask, request, jsonify
from huggingface_hub import login
from transformers import AutoTokenizer, pipeline
import torch

# Function to generate text using Llama model
def generate_text(prompt):
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

# Flask app initialization
app = Flask(__name__)

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

# Route to handle POST requests for text generation
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    generated_text = generate_text(prompt)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
 