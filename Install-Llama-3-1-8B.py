from unsloth import FastLanguageModel

from utilFunctions import clearGGUFDir

'Funktion zum Retraining des Modells mit neuen Daten'
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"unsloth/Meta-Llama-3.1-8B-Instruct",
    dtype = None,
    load_in_4bit = True,
    max_seq_length = 2048,
)

MODEL_PATH="models/Llama-3.1-8B-Instruct"
GGUF_PATH=f"{MODEL_PATH}/gguf"

# Save it as GGUF (quantized)
model.save_pretrained_gguf(GGUF_PATH, tokenizer, quantization_method = "q4_k_m")

# Clear GGUF-directory except quantization file
clearGGUFDir(GGUF_PATH)

# Save model configuration and tokenizer 
model.save_pretrained(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)