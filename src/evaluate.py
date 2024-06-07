import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from sklearn.metrics import mean_squared_error
from config import MODEL_NAME, OUTPUT_DIR

# Load dataset
df = pd.read_csv('../dataset/big_five_dataset.csv')
val_df = df.sample(frac=0.2)

# Convert to Hugging Face Dataset
val_dataset = Dataset.from_pandas(val_df)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

val_dataset = val_dataset.map(tokenize_function, batched=True)

# Trainer
trainer = Trainer(
    model=model,
)

# Evaluate model
predictions = trainer.predict(val_dataset)
preds = predictions.predictions.argmax(-1)
labels = val_dataset['labels']

# Calculate mean squared error
mse = mean_squared_error(labels, preds)
print(f"Mean Squared Error: {mse}")