import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, get_inverse_sqrt_schedule

from torch.optim import Adafactor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from t5.config import T5Config
from t5.span_corruption import *
from t5.encoder_decoder import *


def create_t5_pretraining_data(input_token_ids, max_seq_length=512, corruption_rate=0.15, mean_span_length=3):
    batch_encoder_inputs, batch_decoder_inputs, labels = [], [], []
    batch_size = input_token_ids.shape[0]
    noise_masks = torch.stack([create_span_corruption_mask(max_seq_length, corruption_rate, mean_span_length, input_token_ids.device) for _ in range(batch_size)])

    for i in range(batch_size):
        single_token_ids = input_token_ids[i]
        single_noise_mask = noise_masks[i]

        # 1. create single encoder input, decoder input, label
        if i == 0: # for applying the pad sequence
            encoder_sequence = convert_t5_input_format(single_token_ids, single_noise_mask) # encoder_input.shape: [encoder_input_length (not 512)]
            num_pad_1 = max_seq_length - encoder_sequence.shape[0]
            encoder_input_pad_tokens = torch.tensor([50258]*num_pad_1, dtype=torch.long).to(input_token_ids.device) # pad_token_id: 50258
            encoder_input = torch.concat([encoder_sequence, encoder_input_pad_tokens], dim=0)
            batch_encoder_inputs.append(encoder_input)

            decoder_sequence = convert_t5_target_format(single_token_ids, single_noise_mask)
            sos_token = torch.tensor([50257]).to(input_token_ids.device)  # sos_token id: 50257
            eos_token = torch.tensor([50256]).to(input_token_ids.device)  # eos_token id: 50256
            num_pad_2 = max_seq_length - decoder_sequence.shape[0] - 1  # sos/eos token
            decoder_pad_tokens = torch.tensor([50258]*num_pad_2, dtype=torch.long).to(input_token_ids.device)
            decoder_input = torch.concat([sos_token, decoder_sequence, decoder_pad_tokens], dim=0)
            batch_decoder_inputs.append(decoder_input)

            label = torch.concat([decoder_sequence, eos_token, decoder_pad_tokens], dim=0)
            labels.append(label)

        else:
            encoder_sequence = convert_t5_input_format(single_token_ids, single_noise_mask)
            batch_encoder_inputs.append(encoder_sequence)

            decoder_sequence = convert_t5_target_format(single_token_ids, single_noise_mask)
            sos_token = torch.tensor([50257]).to(input_token_ids.device)
            eos_token = torch.tensor([50256]).to(input_token_ids.device)
            decoder_input = torch.concat([sos_token, decoder_sequence], dim=0)
            batch_decoder_inputs.append(decoder_input)
            label = torch.concat([decoder_sequence, eos_token], dim=0)
            labels.append(label)

    # padding
    padded_encoder_sequence = pad_sequence(batch_encoder_inputs, batch_first=True, padding_value=50258).to(torch.long)
    padded_decoder_sequence = pad_sequence(batch_decoder_inputs, batch_first=True, padding_value=50258).to(torch.long)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=50258).to(torch.long)

    return padded_encoder_sequence, padded_decoder_sequence, padded_labels

def init_weights(module, initializer_range=0.02):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=initializer_range)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=initializer_range)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, dataloader, criterion, optimizer, lr_scheduler, scaler, tokenizer, memory_usage, config):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc='Training ...'):
        batch = batch.to(config.device)
        src, trg, labels = create_t5_pretraining_data(batch)
        """
        <extra_id> token is more like a signal for blank
        therefore, no loss should be computed for <extra_id>
        otherwise, the model might learn that a specific token always follows <extra_id>
        
        to avoid loss from <extra_id> token, replace these sentinel tokens with pad_idx
        ignore loss for positions corresponding to sentinel tokens
        because, loss function is torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
        """
        sentinel_token_ids = tokenizer.additional_special_tokens_ids

        for token_id in sentinel_token_ids:
            labels[labels == token_id] = config.pad_idx

        optimizer.zero_grad()
        with autocast(enabled=True):
            logits = model(src, trg)
            loss = criterion(logits.reshape(-1, config.vocab_size), labels.reshape(-1))

        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)  # clip = 0.1

        scaler.step(optimizer)

        old_scale = scaler.get_scale()
        scaler.update()
        new_scale = scaler.get_scale()

        if new_scale >= old_scale:
            lr_scheduler.step()
        """
        when using GradScaler, the scaling factor may adjust its value during the first few iterations, 
        sometimes causing inf/NaN gradients
        In such cases, scaler.step will skip the internal optimizer.step()
        this is normal behavior for GradScaler and usually occurs a few times at the start of training, then disappears  
        
        call scheduler.step() only when optimizer.step() was performed
        """
        memory_usage.append(torch.cuda.memory_allocated())

        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval(model, dataloader, criterion, tokenizer, config):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating ...'):
            batch = batch.to(config.device)
            src, trg, labels = create_t5_pretraining_data(batch)

            sentinel_token_ids = tokenizer.additional_special_tokens_ids

            for token_id in sentinel_token_ids:
                labels[labels == token_id] = config.pad_idx

            logits = model(src, trg)
            loss = criterion(logits.reshape(-1, config.vocab_size), labels.reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

if __name__ == "__main__":
    config = T5Config()

    current_directory = os.getcwd()
    tokenizer = AutoTokenizer.from_pretrained(current_directory+'\my_gpt2_tokenizer')

    ds = load_dataset('hyunjaehyun/token_ids_dataset_for_t5_pretraining')
    train_ds, valid_ds = ds['train'], ds['valid']

    format = {'type': 'torch', 'format_kwargs': {'dtype': torch.long}}
    train_ds.set_format(**format)
    valid_ds.set_format(**format)

    train_dataloader = DataLoader(train_ds['text'], batch_size=config.batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_ds['text'], batch_size=config.batch_size, shuffle=False, drop_last=True)

    model = T5Transformer(config).to(config.device)
    model.apply(init_weights)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)
    optimizer = Adafactor(model.parameters(), lr=config.lr)
    lr_scheduler = get_inverse_sqrt_schedule(optimizer=optimizer, num_warmup_steps=config.warmup_steps)
    scaler = GradScaler(enabled=True)

    memory_usage = []
    best_loss = float('inf')
    patience_check, patience_limit = 0, 8
    train_loss_list, valid_loss_list = [], []
    for epoch in range(config.epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, lr_scheduler, scaler, tokenizer, memory_usage, config)
        print(f'Train Loss: {train_loss:.4f} | Train PPL: {np.exp(train_loss):.2f}')
        valid_loss = eval(model, valid_dataloader, criterion, tokenizer, config)
        print(f'Valid Loss: {valid_loss:.4f} | Valid PPL: {np.exp(valid_loss):.2f}')

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        if valid_loss >= best_loss:  # if validation loss not improved
            patience_check += 1 # patience +1

            if patience_check >= patience_limit: break

        else: # if validation loss improved
            best_loss = valid_loss # best loss = valid loss and,
            patience_check = 0 # reset patience
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict()}, 't5_pretraining_best_model.pt')
            print(f'New model saved with validation loss: {best_loss:.4f}')

    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss,
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict()}, 'last_epoch_model.pt')

    print(f'avg memory usage: {np.mean(memory_usage)}')
    print(f'memory usage peak: {np.max(memory_usage)}')

    plt.plot(train_loss_list, label='train loss')
    plt.plot(valid_loss_list, label='valid loss')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)

    plt.show()






