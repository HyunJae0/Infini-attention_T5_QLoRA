import torch
import torch.nn as nn
import torch.nn.init as init

from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm.auto import tqdm

from t5.encoder_decoder import T5Transformer
from t5.config import T5Config

class ClassificationHead(nn.Module):
    def __init__(self, config, num_labels: int = 3):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout_rate)
        self.dense = nn.Linear(config.d_model, config.d_model)
        self.classifier = nn.Linear(config.d_model, num_labels)

        init.trunc_normal_(self.dense.weight, std=0.02)
        init.trunc_normal_(self.classifier.weight, std=0.02)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.classifier(hidden_states)

class EncoderOnlyClassifier(nn.Module):
    def __init__(self, config, num_labels: int = 3):
        super().__init__()
        self.transformer = T5Transformer(config)
        self.classification_head = ClassificationHead(config, num_labels=num_labels)
        self.pad_idx = config.pad_idx

    def forward(self, input_ids):
        src_mask = self.transformer.create_encoder_mask(input_ids)  
        encoder_output = self.transformer.Encoder(input_ids, src_mask)  

        m = (input_ids != self.pad_idx).unsqueeze(-1).type_as(encoder_output)
        sentence_representation = (encoder_output * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0) 
        logits = self.classification_head(sentence_representation)
        return logits
    

def train(model, loader, criterion, optimizer, scheduler, config):
    model.train()
    total_loss = 0.0
    correct, count = 0, 0

    for batch in tqdm(loader, desc="Train"):
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device).view(-1)  # [B]

        optimizer.zero_grad()

        logits = model(input_ids)          # [B,3]
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=-1).view(-1)
        correct += (preds==labels).sum().item()
        count += labels.numel()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        
    total_acc = correct / count
    return total_loss / len(loader), total_acc

@torch.no_grad()
def evaluate(model, loader, criterion, config):
    model.eval()
    total_loss = 0.0
    correct, count = 0, 0

    for batch in tqdm(loader, desc="Eval"):
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device).view(-1)

        logits = model(input_ids)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=-1).view(-1)
        correct += (preds==labels).sum().item()
        count += labels.numel()

        total_loss += loss.item()
        
    total_acc = correct / count
    return total_loss / len(loader), total_acc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    config = T5Config()

    EPOCHS = 15
    BATCH_SIZE = 64
    LR = 2e-5 

    ds = load_dataset("hyunjaehyun/token_ids_dataset_for_t5_finetuning2")
    train_ds, valid_ds, test_ds = ds["train"], ds["valid"], ds["test"]

    fmt = {"type": "torch", "format_kwargs": {"dtype": torch.long}}
    train_ds.set_format(**fmt)
    valid_ds.set_format(**fmt)
    test_ds.set_format(**fmt)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = EncoderOnlyClassifier(config).to(config.device)
    check_point = torch.load("last_epoch_model.pt", map_location=config.device, weights_only=True)
    model.transformer.load_state_dict(check_point["model_state_dict"])
    print(f'The model has {count_parameters(model):,} trainable parameters')

    criterion = nn.CrossEntropyLoss()  
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = total_steps // 10  # 10%
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_valid_acc = 0.0
    patience_check, patience_limit = 0, 3
    best_path = "best_finetuning_model.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, scheduler, config)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, config)

        print(f"[Epoch {epoch}/{EPOCHS}] "
              f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
              f"valid_loss: {valid_loss:.4f} valid_acc: {valid_acc:.4f}")

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            patience_check = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_valid_acc": best_valid_acc
            }, best_path)
            print(f"saved: {best_path} (best_valid_acc: {best_valid_acc:.4f})")
        else:
            patience_check += 1
            if patience_check == patience_limit: break

    best = torch.load(best_path, map_location=config.device)
    model.load_state_dict(best["model_state_dict"])

    test_loss, test_acc = evaluate(model, test_loader, criterion, config)
    print(f"[Test] loss={test_loss:.4f} acc={test_acc:.4f}")

