# %%
from datasets import load_dataset, load_metric
from transformers import (
    BertTokenizer,
    BertForQuestionAnswering,
    Trainer,
    TrainingArguments,
)

# %%
data = load_dataset("squad")

# %%
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
squad_metric = load_metric("squad")
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")

training_args = TrainingArguments(
    "bert_squad", evaluation_strategy="epoch", no_cuda=True
)

trainer = Trainer(
    model=model,
    args=TrainingArguments,
    tokenizer=tokenizer,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    compute_metrics=squad_metric,
)

# %%
trainer.evaluate()
