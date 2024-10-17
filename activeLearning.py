from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import json

class TrainingRequest(BaseModel):
    feedback: list
    provider: dict

router = APIRouter()

@router.post("/retrain")
async def retrain_model(request: TrainingRequest):
    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
        from utilFunctions import clearGGUFDir, clearMainDir
        from transformers import TrainingArguments
        from datasets import Dataset
        from trl import SFTTrainer
        import pandas as pd

        model_name=request.provider["data"]["model"].split('/')[0]
        model_path=f"models/{model_name}"

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_path,
            dtype = None,
            load_in_4bit = True,
            max_seq_length = 2048
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            use_rslora = False,
            loftq_config = None,
        )

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
                num_train_epochs = 3,
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

        if "finetuned" in model_path: trainer_eval_stats = trainer.evaluate()

        trainer_train_stats = trainer.train()

        if "finetuned" not in model_path: model_path += "-finetuned"
        # Save model as quantized GGUF-file for hosting purposes
        model.save_pretrained_gguf(
            save_directory=model_path + "/gguf", 
            tokenizer=tokenizer, 
            quantization_method="q4_k_m"
        )
        
        # Clear GGUF-directory except quantization file
        await clearGGUFDir(model_path + "/gguf")

        # Save model configuration and tokenizer for future finetuning
        model.save_pretrained(model_path), tokenizer.save_pretrained(model_path)

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

        del model_name
        del model_path
        del model
        del tokenizer
        del dataset
        del alpaca_prompt
        del formatting_prompts_func

        gc.collect()
        torch.cuda.empty_cache()

        clearMainDir()