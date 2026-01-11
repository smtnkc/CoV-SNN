import os
import torch
import random
import numpy as np
from transformers import RobertaConfig, RobertaForMaskedLM, Trainer, TrainingArguments
from torch.utils.data import DataLoader, TensorDataset, random_split
import time
from datetime import timedelta

os.environ["TOKENIZERS_PARALLELISM"] = "false"
np.set_printoptions(threshold=2000)
torch.set_printoptions(threshold=2000)
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print(torch.cuda.is_available())

tds = torch.load("../data/tensor_dataset_12M.pth")

# Define a custom data collator
class CustomDataCollator:
    def __call__(self, features):
        input_ids = torch.stack([f[0] for f in features])
        attention_mask = torch.stack([f[1] for f in features])
        labels = torch.stack([f[2] for f in features])
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels}

# Initialize the custom data collator
custom_dc = CustomDataCollator()

print(len(tds))

# Define the ratio for train/test split
train_size = int(0.8 * len(tds))
eval_size = len(tds) - train_size

# Split the dataset
train_dataset, eval_dataset = random_split(tds, [train_size, eval_size])
print(len(train_dataset), len(eval_dataset))

EMBED_SIZE = 128
model_out_dir = f"../mlm_checkpoints/CoV-RoBERTa_{EMBED_SIZE}"
if not os.path.exists(model_out_dir):
    os.makedirs(model_out_dir)

# https://huggingface.co/docs/transformers/model_doc/roberta#transformers.RobertaConfig

config = RobertaConfig(
    vocab_size=10000, # defaults to 50265
    hidden_size=768, # defaults to 768
    max_position_embeddings=EMBED_SIZE, # defaults to 512
    num_attention_heads=12, # defaults to 12
    num_hidden_layers=6, # defaults to 12
    type_vocab_size=1 # defaults to 2
)

model = RobertaForMaskedLM(config=config)

print(model.num_parameters())

# https://huggingface.co/docs/transformers/v4.35.0/en/main_classes/trainer#transformers.TrainingArguments

training_args = TrainingArguments(
    report_to = 'tensorboard',
    optim='adamw_torch',
    output_dir=model_out_dir,
    evaluation_strategy = 'steps', 
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=512, # 10168246/512 = 19765 steps
    per_device_eval_batch_size=512,  # 2542062/512  = 4956  steps
    save_steps=1000,
    save_total_limit=2,
    logging_steps=1000,
    prediction_loss_only=True,
    push_to_hub=False,
    seed=42,
    data_seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=custom_dc,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

start_time = time.time()

trainer.train()

elapsed_time = time.time() - start_time
formatted_time = str(timedelta(seconds=elapsed_time))

print(f"Elapsed time: {formatted_time}")

trainer.evaluate()

trainer.save_model()
