import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report

from common import EXP_ROOT, load_json

os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
MODEL_NAME = 'allenai/scibert_scivocab_uncased'
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 8
SEED = 42
set_seed(SEED)
torch.manual_seed(SEED)


class SciBERTCompoundDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        encoded = self.tokenizer(sample['text'], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded['labels'] = torch.tensor(self.label2id[sample['output']], dtype=torch.long)
        return encoded


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def narrative_key(sample):
    return {'compound': sample['compound'], 'space_group': int(sample['space_group']), 'label': sample['output']}


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {'accuracy': accuracy_score(labels, preds), 'f1': f1_score(labels, preds, average='weighted')}


def resolve_split_dir(split_name: str):
    return EXP_ROOT / 'scibert_finetuning' / split_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-name', required=True)
    args = parser.parse_args()

    split_dir = resolve_split_dir(args.split_name)
    json_path = split_dir / 'finetune_dataset_scibert_improved.json'
    split_manifest_path = split_dir / 'split_manifest.json'
    output_dir = split_dir / 'scibert-finetuned-weighted-improved-input'
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(json_path)
    manifest = load_json(split_manifest_path)
    dataset_keys = [narrative_key(row) for row in data]
    if dataset_keys != manifest['record_keys']:
        raise ValueError('SciBERT split dataset does not match split manifest')

    unique_labels = sorted(set(d['output'] for d in data))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = SciBERTCompoundDataset(data, tokenizer, label2id, max_length=MAX_LENGTH)
    trainable_labels = [label2id[row['output']] for row in data]
    train_indices = np.array(manifest['train_indices'])
    val_indices = np.array(manifest['validation_indices'])
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_labels = np.array([trainable_labels[i] for i in train_indices])
    class_counts = np.bincount(train_labels, minlength=len(label2id))
    class_weights = len(train_labels) / (len(label2id) * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    (output_dir / 'label2id.json').write_text(json.dumps(label2id, indent=2))
    (output_dir / 'id2label.json').write_text(json.dumps(id2label, indent=2))
    (output_dir / 'class_weights.json').write_text(json.dumps({
        'class_counts': dict(zip(unique_labels, class_counts.tolist())),
        'class_weights': dict(zip(unique_labels, class_weights.tolist())),
    }, indent=2))

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(label2id), id2label=id2label, label2id=label2id)
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='epoch',
        save_total_limit=2,
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        seed=SEED,
        report_to='none',
    )
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        class_weights=class_weights_tensor,
    )
    trainer.train()
    best_export_dir = output_dir / 'best_model'
    trainer.save_model(best_export_dir)
    tokenizer.save_pretrained(best_export_dir)

    preds = trainer.predict(val_dataset)
    report = classification_report(preds.label_ids, preds.predictions.argmax(-1), target_names=unique_labels, output_dict=True)
    (output_dir / 'validation_classification_report.json').write_text(json.dumps(report, indent=2))
    print(best_export_dir)


if __name__ == '__main__':
    main()
