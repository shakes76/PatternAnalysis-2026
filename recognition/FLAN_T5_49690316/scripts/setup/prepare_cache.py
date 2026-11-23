from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import MODEL_ID
from dataset import load_audited_data, split_train_development


print("Loading tokeniser...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading pretrained model on CPU...", flush=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

print("Loading and auditing dataset...", flush=True)
_, clean_train = load_audited_data()
training, development = split_train_development(clean_train)

print("Training rows:", len(training), flush=True)
print("Development rows:", len(development), flush=True)
print("Cache preparation completed", flush=True)