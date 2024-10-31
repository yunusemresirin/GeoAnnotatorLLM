from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json

class TrainingRequest(BaseModel):
    feedback: list
    provider: dict

router = APIRouter()

@router.post("/retrain-hf")
async def retrain_model(request: TrainingRequest):
    try:

        import torch
        import pandas as pd
        from trl import SFTTrainer
        from datasets import Dataset
        from transformers import TrainingArguments
        from peft import LoraConfig, get_peft_model
        from utilFunctions import clearMainDir, get_next_version
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name=request.provider['data']['model'].split('/')[0]
        model_path=f"models/{model_name}"

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            use_rslora=False,
            loftq_config=None,
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, lora_config)

        alpaca_prompt = """You are an assitant that strictly extracts geographic references from the input. For each location, provide the place-name (exactly as in the text), the latitude and the longitude of the place as a json-object, like {{ name: place-name, position: [latitude, longitude] }}. Create a json-list out of these objects. In the list, there should be no repetitive places with the same place-name. Please only return the value with no explanation or further information and as a normal text without labeling it as json.

        ### Input:
        {}

        ### Output:
        {}"""

        EOS_TOKEN = tokenizer.eos_token
        def formatting_prompts_func(data):
            inputs       = data["text"]
            outputs      = data["corrections"]
            texts = []
            for input, output in zip(inputs, outputs):
                output_str = json.dumps(output, indent=2)
                text = alpaca_prompt.format(input, output_str) + EOS_TOKEN
                texts.append(text)
            return { "text" : texts, }
        pass

        dataset = Dataset.from_pandas(df=pd.DataFrame(request.feedback))
        dataset = dataset.map(formatting_prompts_func, batched=True, load_from_cache_file=False)

        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            dataset_text_field = "text", 
            train_dataset = dataset,
            eval_dataset = dataset,
            max_seq_length = 2048,
            dataset_num_proc = 2,
            packing = False,
            args = TrainingArguments(
                per_device_eval_batch_size = 2,
                do_eval=False,
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                num_train_epochs = 2,
                learning_rate = 2e-4,
                fp16 = True,
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs",
            )
        )

        trainer_eval_stats = trainer.evaluate() if "finetuned" in model_path else None

        trainer_train_stats = trainer.train()

        if "finetuned" not in model_name: model_path += "-finetuned-1"
        else:
            version = await get_next_version(model_name)
            model_name = f"{'-'.join(model_name.split('-')[:-1])}-{version}"
            model_path = f"models/{model_name}"
        # Save model configuration and tokenizer for future finetuning
        model.merge_and_unload(), model.save_pretrained(model_path), tokenizer.save_pretrained(model_path)

        # Conversion to GGUF-format
        # Using llama.cpp

        return {
            "message": "Finetuning process completed. Model has been saved.",
            "trainer_train_stats": trainer_train_stats[2],
            "trainer_eval_stats": trainer_eval_stats or None,
            "trainer_args": trainer.args
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain-Job failed: {str(e)}")
    finally:
        # Clear cache and collect garbage
        import gc
        import torch

        del model_name
        del model_path
        del model
        del tokenizer
        del dataset
        del alpaca_prompt
        del formatting_prompts_func

        gc.collect() 
        torch.cuda.empty_cache()

        await clearMainDir()

        print("Clearing complete.")

@router.post("/retrain")
async def retrain_model(request: TrainingRequest):
    try:
        import pandas as pd
        from trl import SFTTrainer
        from datasets import Dataset
        from transformers import TrainingArguments
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from utilFunctions import clearGGUFDir, clearMainDir, get_next_version

        model_name:str = request.provider['data']['model'].split('/')[0]
        model_path = f"models/{model_name}"

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            dtype = None,
            load_in_4bit = True,
            max_seq_length = 2048
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                              "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0.1,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            use_rslora = False,
            loftq_config = None,
        )

        alpaca_prompt = """You are an assitant that strictly extracts geographic references from the user-input. For each location, provide the place-name (exactly as mentioned in the user-input), the latitude and the longitude of the place as a json-object, like {{ name: place-name, position: [latitude, longitude] }}. Create a json-list out of these objects. In the list, there should be no repetitive places with the same place-name. Do not extract or provide places that are not mentioned in the user-input. The positions should be as precise as possible. Please only return the json-string with no explanation or further information and as a normal text without labeling it as json.

        ### Input:
        {}

        ### Output:
        {}"""

        EOS_TOKEN = tokenizer.eos_token
        def formatting_prompts_func(data):
            inputs       = data["text"]
            outputs      = data["corrections"]
            texts = []
            for input, output in zip(inputs, outputs):
                output_str = json.dumps(output, indent=2)
                text = alpaca_prompt.format(input, output_str) + EOS_TOKEN
                texts.append(text)
            return { "text" : texts }
        pass

        dataset = Dataset.from_pandas(df=pd.DataFrame(request.feedback))
        dataset = dataset.map(formatting_prompts_func, batched = True, load_from_cache_file=False)

        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            dataset_text_field = "text",
            train_dataset = dataset,
            eval_dataset = dataset,
            max_seq_length = 2048,
            dataset_num_proc = 2,
            packing = False,
            args = TrainingArguments(
                per_device_eval_batch_size = 2,
                do_eval=False,
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                num_train_epochs = 2,
                learning_rate = 2e-4,
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = "outputs"
            )
        )

        trainer_eval_stats = trainer.evaluate() if "finetuned" in model_name else None

        trainer_train_stats = trainer.train()

        if "finetuned" not in model_name: model_path += "-finetuned-1"
        else:
            version = await get_next_version(model_name)
            model_name = f"{'-'.join(model_name.split('-')[:-1])}-{version}"
            model_path = f"models/{model_name}"
        # Save model configuration and tokenizer for future finetuning
        model.save_pretrained(model_path), tokenizer.save_pretrained(model_path)
        # Save model as quantized GGUF-file for hosting purposes
        model.save_pretrained_gguf(
            save_directory=model_path + "/gguf", 
            tokenizer=tokenizer, 
            quantization_method="q4_k_m"
        )
        
        # Clear GGUF-directory except quantization file
        await clearGGUFDir(model_path + "/gguf")

        return {
            "message": "Finetuning process completed. Model has been saved.",
            "trainer_train_stats": trainer_train_stats[2],
            "trainer_eval_stats": trainer_eval_stats,
            "trainer_args": trainer.args
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain-Job failed: {str(e)}")
    finally:
        # Clear cache and collect garbage
        import gc
        import torch

        if model_name: del model_name
        if model_path: del model_path
        if model: del model
        if tokenizer: del tokenizer
        if dataset: del dataset
        if alpaca_prompt: del alpaca_prompt
        if formatting_prompts_func: del formatting_prompts_func

        gc.collect() 
        torch.cuda.empty_cache()

        await clearMainDir()

        print("Clearing complete.")