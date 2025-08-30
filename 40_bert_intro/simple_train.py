"""
Simplified training script to ensure it works correctly.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
import yaml
from tqdm import tqdm
import json

from model.bert_encoder import BERTModel
from model.heads import BERTForPreTraining, compute_mlm_accuracy
from tokenization.simple_wp import SimpleTokenizer


def create_simple_dataset(texts, tokenizer, max_seq_length=64, mlm_prob=0.15):
    """Create a simple dataset with fixed-size examples."""
    examples = []
    
    for text in texts:
        # Tokenize
        tokens = tokenizer.tokenize(text)
        
        # Truncate and add special tokens
        max_tokens = max_seq_length - 2
        tokens = tokens[:max_tokens]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
        
        # Pad to exact length
        while len(tokens) < max_seq_length:
            tokens.append(tokenizer.pad_token)
            
        # Convert to IDs
        input_ids = [tokenizer.vocab.get(token, tokenizer.vocab[tokenizer.unk_token]) 
                    for token in tokens]
        
        # Create segment IDs (all 0 for single sentence)
        segment_ids = [0] * max_seq_length
        
        # Create attention mask
        attention_mask = [1 if token != tokenizer.pad_token else 0 for token in tokens]
        
        # Apply MLM masking
        masked_ids = input_ids.copy()
        mlm_labels = [-100] * max_seq_length
        
        special_tokens = {tokenizer.vocab[tokenizer.cls_token], 
                         tokenizer.vocab[tokenizer.sep_token], 
                         tokenizer.vocab[tokenizer.pad_token]}
        
        for i, token_id in enumerate(input_ids):
            if token_id not in special_tokens and torch.rand(1).item() < mlm_prob:
                mlm_labels[i] = token_id
                
                rand = torch.rand(1).item()
                if rand < 0.8:
                    masked_ids[i] = tokenizer.vocab[tokenizer.mask_token]
                elif rand < 0.9:
                    masked_ids[i] = torch.randint(5, len(tokenizer.vocab), (1,)).item()
        
        examples.append({
            'input_ids': torch.tensor(masked_ids),
            'segment_ids': torch.tensor(segment_ids),
            'attention_mask': torch.tensor(attention_mask),
            'mlm_labels': torch.tensor(mlm_labels)
        })
    
    return examples


def train_simple():
    # Simple config
    config = {
        'vocab_size': 1000,
        'd_model': 128,
        'n_layers': 2,
        'n_heads': 2,
        'd_ff': 256,
        'max_seq_length': 64,
        'dropout': 0.1,
        'use_nsp': False,
        'weight_tie_mlm': True
    }
    
    # Load corpus
    with open('data/toy_corpus.txt', 'r') as f:
        texts = [line.strip() for line in f if line.strip()]
        
    print(f"Loaded {len(texts)} texts")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config['vocab_size'])
    tokenizer.build_vocab(texts)
    config['vocab_size'] = tokenizer.get_vocab_size()
    
    print(f"Vocab size: {config['vocab_size']}")
    
    # Create dataset
    dataset = create_simple_dataset(texts, tokenizer, config['max_seq_length'])
    print(f"Created {len(dataset)} examples")
    
    # Verify all examples have the same shape
    for i, example in enumerate(dataset[:5]):
        print(f"Example {i}: input_ids {example['input_ids'].shape}")
    
    # Create model
    bert_model = BERTModel(config)
    model = BERTForPreTraining(bert_model, config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create simple dataloader
    batch_size = 8
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Training loop
    model.train()
    
    for epoch in range(2):
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        print(f"Epoch {epoch + 1}")
        
        # Simple batching
        for i in tqdm(range(0, len(dataset), batch_size), desc="Training"):
            batch = dataset[i:i+batch_size]
            
            if len(batch) == 0:
                continue
                
            # Stack tensors
            input_ids = torch.stack([ex['input_ids'] for ex in batch]).to(device)
            segment_ids = torch.stack([ex['segment_ids'] for ex in batch]).to(device)
            attention_mask = torch.stack([ex['attention_mask'] for ex in batch]).to(device)
            mlm_labels = torch.stack([ex['mlm_labels'] for ex in batch]).to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(input_ids, segment_ids, attention_mask, mlm_labels)
            
            loss = outputs['loss']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_acc += compute_mlm_accuracy(outputs['mlm_logits'], mlm_labels)
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}")
    
    # Save model
    os.makedirs('test_run', exist_ok=True)
    torch.save(model.state_dict(), 'test_run/final_model.pt')
    
    with open('test_run/config.json', 'w') as f:
        json.dump(config, f, indent=2)
        
    tokenizer.save_vocab('test_run/vocab.json')
    
    print("Training completed successfully!")
    print(f"Final loss: {avg_loss:.4f}")
    print(f"Final accuracy: {avg_acc:.4f}")
    
    return avg_loss, avg_acc


if __name__ == "__main__":
    train_simple()

