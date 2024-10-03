from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

from utilFunctions import clearGGUFDir
from pydantic import BaseModel
from fastapi import APIRouter
import json

class TrainingRequest(BaseModel):
    feedback: list
    provider: dict

router = APIRouter()

@router.get("/test")
def test():
    return "Success!"

@router.post("/retrain")
def retrain_model(request: TrainingRequest):
    model_name=request.provider["data"]["model"]

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
    def formatting_prompts_func(examples):
        inputs       = examples["text"]
        outputs      = examples["corrections"]
        texts = []
        for input, output in zip(inputs, outputs):
            output_str = json.dumps(output, indent=2)
            text = alpaca_prompt.format(input, output_str) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }
    pass

    dataset = load_dataset("json", data_files="../feedback/test.json", split='train')
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    # Counting data for training steps
    def Datacount():
        try:
            with open("feedback/test.json", 'r') as f:
                data = json.load(f)
            return len(data)
        except:
            return 60

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # num_train_epochs = 2,
            max_steps = Datacount(),
            learning_rate = 2e-4,
            embedding_learning_rate = 2e-5,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    trainer_stats = trainer.train()

    # Modell speichern
    model.save_pretrained_gguf(f"models/{model_name}/gguf", tokenizer, quantization_method = "q4_k_m")
    
    # Clear GGUF-directory except quantization file
    clearGGUFDir(GGUF_PATH)

    # Save model configuration and tokenizer 
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    
    model.save_pretrained_merged(f"models/{model_name}", tokenizer, save_method = "merged_4bit")

    metrics = trainer.args
    return {
        "message": "Finetuning process completed. Model has been saved.",
        "metrics": {
            "training_loss": trainer_stats.training_loss,
            "learning_rate": metrics.learning_rate,
            "train_epochs": metrics.num_train_epochs,
            "train_max_steps": metrics.max_steps
        }
    }