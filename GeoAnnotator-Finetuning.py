from unsloth import FastLanguageModel, is_bfloat16_supported
# from transformers import TrainingArguments
# from datasets import Dataset
# from trl import SFTTrainer
# import pandas as pd

# from utilFunctions import clearGGUFDir, clearMainDir
# import argparse
# import json

# parser = argparse.ArgumentParser(description="Retrain-Job of LLM")
# parser.add_argument('input_path', type=str, help='Input data path')
# args = parser.parse_args()

# with open(args.input_path, 'r') as f:
#     data = json.load(f)

# # Arguments
# feedback = data["feedback"]
# provider = data["provider"]

# model_name=provider["data"]["model"].split('/')[0]

model_name = "Llama-3.1-8B-Instruct-finetuned-1"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = f"models/{model_name}",
    dtype = None,
    load_in_4bit = True,
    max_seq_length = 2048,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0.1,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# alpaca_prompt = """You are an assitant that strictly extracts geographic references from the input. For each location, provide the place-name (exactly as in the text), the latitude and the longitude of the place as a json-object, like {{ name: place-name, position: [latitude, longitude] }}. Create a json-list out of these objects. In the list, there should be no repetitive places with the same place-name. Please only return the value with no explanation or further information and as a normal text without labeling it as json.

# ### Input:
# {}

# ### Output:
# {}"""

# EOS_TOKEN = tokenizer.eos_token
# def formatting_prompts_func(examples):
#     inputs       = examples["text"]
#     outputs      = examples["corrections"]
#     texts = []
#     for input, output in zip(inputs, outputs):
#         output_str = json.dumps(output, indent=2)
#         text = alpaca_prompt.format(input, output_str) + EOS_TOKEN
#         texts.append(text)
#     return { "text" : texts, }
# pass

# # dataset = load_dataset("json", data_files="../feedback/test.json", split='train')
# dataset = Dataset.from_pandas(pd.DataFrame(feedback))
# dataset = dataset.map(formatting_prompts_func, batched = True)
# dataset = dataset.train_test_split(test_size=0.2)

# # TrainingArguments - configurations
# train_dataset = dataset['train']
# eval_dataset = dataset['test']

# # training_steps = len(request.feedback) if request.feedback is not None else 120

# trainer = SFTTrainer(
#     model = model,
#     tokenizer = tokenizer,
#     dataset_text_field = "text",
#     eval_dataset = eval_dataset,
#     train_dataset = train_dataset,
#     max_seq_length = 2048,
#     dataset_num_proc = 2,
#     packing = False,
#     args = TrainingArguments(
#         evaluation_strategy="epoch",
#         per_device_eval_batch_size = 2,
#         per_device_train_batch_size = 2,
#         gradient_accumulation_steps = 4,
#         warmup_steps = 5,
#         num_train_epochs = 3,
#         # max_steps = training_steps,
#         learning_rate = 2e-4,
#         fp16 = not is_bfloat16_supported(),
#         bf16 = is_bfloat16_supported(),
#         logging_steps = 1,
#         optim = "adamw_8bit",
#         weight_decay = 0.01,
#         lr_scheduler_type = "linear",
#         seed = 3407,
#         output_dir = "outputs"
#     )
# )

# trainer_stats = trainer.train()

# # Modell speichern
GGUF_PATH=f"models/{model_name}/gguf"

model.save_pretrained_gguf(GGUF_PATH, tokenizer, quantization_method = "q4_k_m")

# # Clear GGUF-directory except quantization file
# clearGGUFDir(GGUF_PATH)

# # Save model configuration and tokenizer 
# model.save_pretrained(f"models/{model_name}-finetuned")
# tokenizer.save_pretrained(f"models/{model_name}-finetuned")

# clearMainDir()

# import gc
# import torch

# del model
# del model_name
# del dataset
# del formatting_prompts_func
# del alpaca_prompt

# gc.collect()
# torch.cuda.empty_cache()

# hf_token="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"