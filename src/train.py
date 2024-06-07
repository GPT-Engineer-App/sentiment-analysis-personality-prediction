import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from peft import PeftModel, PeftConfig
from lora import LoraConfig, LoraModel
from sklearn.model_selection import train_test_split
from config import MODEL_NAME, OUTPUT_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE

# Load dataset
df = pd.read_csv('../dataset/big_five_dataset.csv')
train_df, val_df = train_test_split(df, test_size=0.2)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=5)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Load PEFT and LoRA configurations
peft_config = PeftConfig()
lora_config = LoraConfig()

# Apply PEFT and LoRA
peft_model = PeftModel(model, peft_config)
lora_model = LoraModel(peft_model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()