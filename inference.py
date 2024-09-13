from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Configure quantization to load the model in 4-bit precision
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    bnb_4bit_compute_dtype=torch.float16,  # Use FP16 for compute precision
    bnb_4bit_quant_type='nf4'  # Choose 'nf4' quantization
)

# Load the model with quantization
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path='./results', 
    quantization_config=bnb_config,
    device_map="auto"  # Automatically use the GPU if available
)

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path='microsoft/Phi-3-mini-4k-instruct',
    trust_remote_code=True
)

pipe = pipeline(task='question-answering', model=model, tokenizer=tokenizer)

prompt = 'What is Paracetamol poisoning and explain in detail?'

answer = pipe(prompt)
print(answer)