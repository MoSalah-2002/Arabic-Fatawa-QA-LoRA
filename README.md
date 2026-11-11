# Arabic Fatawa QA — Qwen3-4B LoRA

Experimental Arabic question-answering fine-tuning using Qwen/Qwen3-4B, TRL SFTTrainer and PEFT LoRA.

## Pipeline
- Dataset: Kaggle `abdallahelsaadany/fatawa`, CSV `88kData.csv` (not included).
- Clean missing/duplicate rows and normalize Arabic/English text.
- Use a 10,000-example subset with a 70/30 train/evaluation split.
- Format question/answer pairs as user/assistant messages.
- LoRA: rank 64, alpha 128, dropout 0.1; target q_proj, k_proj and v_proj.
- One training epoch, learning rate 5e-5.

## Run
Install `requirements.txt`, download the dataset from Kaggle, then run `python train.py` with a suitable GPU. Set `FATAWA_CSV` to the CSV path if needed.

## Limitations
This is an experiment, not a validated religious authority. Outputs may be inaccurate; do not use it to issue rulings or replace qualified scholars. No expert evaluation or benchmark is claimed. Training may require adapting to your installed TRL/Transformers versions.