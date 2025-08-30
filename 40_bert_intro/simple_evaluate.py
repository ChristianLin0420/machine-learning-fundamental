"""
Simple evaluation script that uses the same robust approach as simple_train.py
"""

import os
import torch
import torch.nn.functional as F
import json
from model.bert_encoder import BERTModel
from model.heads import BERTForPreTraining, compute_mlm_accuracy
from tokenization.simple_wp import SimpleTokenizer


def create_simple_eval_dataset(texts, tokenizer, max_seq_length=64, mlm_prob=0.15):
    """Create a simple evaluation dataset with fixed-size examples."""
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
        
        # Apply MLM masking for evaluation
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
                    # Make sure random token is within vocab bounds
                    vocab_size = len(tokenizer.vocab)
                    masked_ids[i] = torch.randint(5, vocab_size, (1,)).item()
        
        examples.append({
            'input_ids': torch.tensor(masked_ids),
            'segment_ids': torch.tensor(segment_ids),
            'attention_mask': torch.tensor(attention_mask),
            'mlm_labels': torch.tensor(mlm_labels),
            'original_text': text
        })
    
    return examples


def evaluate_simple(checkpoint_path, corpus_file, vocab_file):
    """Simple evaluation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, 'config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        print("Config not found, creating default...")
        config = {
            'vocab_size': 147,
            'd_model': 128,
            'n_layers': 2,
            'n_heads': 2,
            'd_ff': 256,
            'max_seq_length': 64,
            'dropout': 0.1,
            'use_nsp': False,
            'weight_tie_mlm': True
        }
    
    print(f"Model config: {config}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {vocab_file}")
    tokenizer = SimpleTokenizer(vocab_file=vocab_file)
    print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
    
    # Update config vocab size to match tokenizer
    config['vocab_size'] = tokenizer.get_vocab_size()
    
    # Load corpus
    with open(corpus_file, 'r') as f:
        texts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(texts)} texts for evaluation")
    
    # Create evaluation dataset
    eval_dataset = create_simple_eval_dataset(texts, tokenizer, config['max_seq_length'])
    print(f"Created {len(eval_dataset)} evaluation examples")
    
    # Check dataset consistency
    for i, example in enumerate(eval_dataset[:3]):
        print(f"Example {i}: input_ids {example['input_ids'].shape}, max_id: {example['input_ids'].max()}")
    
    # Create model
    bert_model = BERTModel(config)
    model = BERTForPreTraining(bert_model, config)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Evaluate
    total_loss = 0
    total_acc = 0
    total_masked_tokens = 0
    num_batches = 0
    
    batch_size = 8
    
    with torch.no_grad():
        for i in range(0, len(eval_dataset), batch_size):
            batch = eval_dataset[i:i+batch_size]
            
            if len(batch) == 0:
                continue
            
            # Stack tensors
            input_ids = torch.stack([ex['input_ids'] for ex in batch]).to(device)
            segment_ids = torch.stack([ex['segment_ids'] for ex in batch]).to(device)
            attention_mask = torch.stack([ex['attention_mask'] for ex in batch]).to(device)
            mlm_labels = torch.stack([ex['mlm_labels'] for ex in batch]).to(device)
            
            print(f"Batch {num_batches}: input_ids shape {input_ids.shape}, max_id: {input_ids.max()}, vocab_size: {config['vocab_size']}")
            
            # Ensure no token IDs exceed vocab size
            if input_ids.max() >= config['vocab_size']:
                print(f"Warning: Token ID {input_ids.max()} exceeds vocab size {config['vocab_size']}")
                input_ids = torch.clamp(input_ids, 0, config['vocab_size'] - 1)
            
            try:
                # Forward pass
                outputs = model(input_ids, segment_ids, attention_mask, mlm_labels)
                
                loss = outputs['loss']
                total_loss += loss.item()
                
                # Compute accuracy
                acc = compute_mlm_accuracy(outputs['mlm_logits'], mlm_labels)
                total_acc += acc
                
                # Count masked tokens
                masked_tokens = (mlm_labels != -100).sum().item()
                total_masked_tokens += masked_tokens
                
                print(f"Batch {num_batches}: Loss = {loss.item():.4f}, Accuracy = {acc:.4f}, Masked tokens = {masked_tokens}")
                
                num_batches += 1
                
            except Exception as e:
                print(f"Error in batch {num_batches}: {e}")
                print(f"Input shapes: {input_ids.shape}, {segment_ids.shape}, {attention_mask.shape}, {mlm_labels.shape}")
                raise
    
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches
        
        print(f"\n=== Evaluation Results ===")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average MLM Accuracy: {avg_acc:.4f}")
        print(f"Total Masked Tokens: {total_masked_tokens}")
        print(f"Batches Processed: {num_batches}")
        
        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'masked_tokens': total_masked_tokens,
            'batches': num_batches
        }
    else:
        print("No batches were successfully processed!")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--corpus', required=True, help='Path to evaluation corpus')
    parser.add_argument('--vocab', required=True, help='Path to vocabulary file')
    
    args = parser.parse_args()
    
    results = evaluate_simple(args.checkpoint, args.corpus, args.vocab)
    
    if results:
        print(f"\nEvaluation completed successfully!")
        print(f"Final MLM Accuracy: {results['accuracy']:.2%}")
    else:
        print("Evaluation failed!")

