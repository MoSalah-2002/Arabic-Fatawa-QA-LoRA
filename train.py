"""Experimental Qwen3-4B LoRA SFT. Expected CSV columns: ques, ans."""
import os, re
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen3-4B"

def clean(value):
    value = str(value)
    value = re.sub(r"[^a-zA-Z\s0-9\u0600-\u06FF]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def main():
    csv_path = os.getenv("FATAWA_CSV", "88kData.csv")
    df = pd.read_csv(csv_path).drop(columns=["ID"], errors="ignore").dropna().drop_duplicates()
    df["ques"], df["ans"] = df["ques"].map(clean), df["ans"].map(clean)
    df = df[df["ans"].str.len() >= 118][["ques", "ans"]].head(10_000)
    ds = Dataset.from_pandas(df.reset_index(drop=True)).train_test_split(test_size=0.3, seed=42)

    def to_chat(row):
        return {"messages": [{"role": "user", "content": row["ques"]},
                             {"role": "assistant", "content": row["ans"]}]}
    ds = ds.map(to_chat, remove_columns=ds["train"].column_names)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto", device_map="auto")
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj"], bias="none"))
    args = SFTConfig(output_dir="./arabic-fatawa-lora", num_train_epochs=1,
        per_device_train_batch_size=6, per_device_eval_batch_size=6,
        learning_rate=5e-5, lr_scheduler_type="linear", warmup_steps=100,
        logging_steps=10, save_strategy="epoch", eval_strategy="epoch",
        load_best_model_at_end=True)
    trainer = SFTTrainer(model=model, args=args, train_dataset=ds["train"],
        eval_dataset=ds["test"], processing_class=tokenizer)
    trainer.train()
    trainer.save_model("./arabic-fatawa-lora")
    tokenizer.save_pretrained("./arabic-fatawa-lora")

if __name__ == "__main__":
    main()
