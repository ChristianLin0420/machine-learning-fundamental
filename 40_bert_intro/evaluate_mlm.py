"""
Evaluation script for BERT MLM pretraining.
Computes MLM accuracy, perplexity, and other metrics.
"""

import os
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging

# Local imports
from model.bert_encoder import BERTModel
from model.heads import BERTForPreTraining, compute_mlm_accuracy, compute_nsp_accuracy
from tokenization.simple_wp import SimpleTokenizer
from tokenization.dataset_mlm import create_mlm_dataloader, load_corpus


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_model_and_config(checkpoint_path, device):
    """Load model and config from checkpoint."""
    # Try to load config
    checkpoint_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(checkpoint_dir, 'config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default config if not found
        config = {
            'vocab_size': 147,  # Match the actual trained model
            'd_model': 128,     # Match BERT-tiny config
            'n_layers': 2,
            'n_heads': 2,
            'd_ff': 256,
            'max_seq_length': 64,  # Match training config
            'dropout': 0.1,
            'use_nsp': False,
            'use_sop': False,
            'weight_tie_mlm': True
        }
        print("Warning: config.json not found, using BERT-tiny default config")
    
    # Load model
    bert_model = BERTModel(config)
    model = BERTForPreTraining(bert_model, config)
    
    # Load checkpoint
    try:
        if checkpoint_path.endswith('.pt'):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Model architecture:")
        print(f"  vocab_size: {config['vocab_size']}")
        print(f"  d_model: {config['d_model']}")
        print(f"  max_seq_length: {config['max_seq_length']}")
        raise
    
    model = model.to(device)
    model.eval()
    
    return model, config


def compute_perplexity(logits, labels):
    """
    Compute perplexity for masked language modeling.
    
    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len) - labels with -100 for non-masked tokens
    
    Returns:
        perplexity: float
    """
    # Only consider positions where labels != -100 (masked positions)
    mask = (labels != -100)
    if mask.sum() == 0:
        return float('inf')
    
    # Add bounds checking for labels
    vocab_size = logits.size(-1)
    valid_labels = labels.clone()
    
    # Clamp labels to valid range and mask out invalid ones
    invalid_mask = (valid_labels >= vocab_size) | (valid_labels < 0)
    if invalid_mask.any():
        print(f"Warning: Found {invalid_mask.sum()} invalid labels (max: {valid_labels.max()}, vocab_size: {vocab_size})")
        valid_labels = torch.clamp(valid_labels, 0, vocab_size - 1)
        mask = mask & ~invalid_mask
    
    if mask.sum() == 0:
        return float('inf')
    
    # Get log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Get log probabilities for correct tokens (only where mask is True)
    masked_labels = valid_labels.masked_fill(~mask, 0)  # Fill non-masked positions with 0
    correct_log_probs = log_probs.gather(dim=-1, index=masked_labels.unsqueeze(-1)).squeeze(-1)
    
    # Mask out non-masked positions
    correct_log_probs = correct_log_probs * mask.float()
    
    # Compute average negative log likelihood
    avg_nll = -correct_log_probs.sum() / mask.sum().float()
    
    # Perplexity is exp of average negative log likelihood
    perplexity = torch.exp(avg_nll).item()
    
    return perplexity


def evaluate_model_detailed(model, dataloader, device, logger, tokenizer=None):
    """Detailed evaluation of the model."""
    model.eval()
    
    all_results = {
        'total_loss': 0,
        'mlm_loss': 0,
        'nsp_loss': 0,
        'mlm_accuracy': 0,
        'nsp_accuracy': 0,
        'perplexity': 0,
        'total_samples': 0,
        'total_masked_tokens': 0
    }
    
    # For detailed analysis
    token_level_results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mlm_labels = batch['mlm_labels'].to(device)
            
            # Forward pass
            if 'nsp_labels' in batch:
                nsp_labels = batch['nsp_labels'].to(device)
                outputs = model(input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels)
            else:
                outputs = model(input_ids, segment_ids, attention_mask, mlm_labels)
            
            batch_size = input_ids.size(0)
            all_results['total_samples'] += batch_size
            
            # Total loss
            all_results['total_loss'] += outputs['loss'].item() * batch_size
            
            # MLM metrics
            if 'mlm_loss' in outputs:
                all_results['mlm_loss'] += outputs['mlm_loss'].item() * batch_size
            
            mlm_acc = compute_mlm_accuracy(outputs['mlm_logits'], mlm_labels)
            all_results['mlm_accuracy'] += mlm_acc * batch_size
            
            # Perplexity
            batch_perplexity = compute_perplexity(outputs['mlm_logits'], mlm_labels)
            if not np.isinf(batch_perplexity):
                masked_tokens = (mlm_labels != -100).sum().item()
                all_results['perplexity'] += batch_perplexity * masked_tokens
                all_results['total_masked_tokens'] += masked_tokens
            
            # NSP metrics
            if 'nsp_logits' in outputs and 'nsp_labels' in batch:
                if 'nsp_loss' in outputs:
                    all_results['nsp_loss'] += outputs['nsp_loss'].item() * batch_size
                nsp_acc = compute_nsp_accuracy(outputs['nsp_logits'], nsp_labels)
                all_results['nsp_accuracy'] += nsp_acc * batch_size
            
            # Collect token-level results for analysis (first few batches)
            if batch_idx < 3 and tokenizer is not None:
                token_level_results.extend(
                    analyze_batch_predictions(batch, outputs, tokenizer)
                )
    
    # Compute averages
    total_samples = all_results['total_samples']
    results = {
        'total_loss': all_results['total_loss'] / total_samples,
        'mlm_accuracy': all_results['mlm_accuracy'] / total_samples,
    }
    
    if all_results['mlm_loss'] > 0:
        results['mlm_loss'] = all_results['mlm_loss'] / total_samples
    
    if all_results['total_masked_tokens'] > 0:
        results['perplexity'] = all_results['perplexity'] / all_results['total_masked_tokens']
    
    if all_results['nsp_loss'] > 0:
        results['nsp_loss'] = all_results['nsp_loss'] / total_samples
        results['nsp_accuracy'] = all_results['nsp_accuracy'] / total_samples
    
    # Log results
    logger.info(f"Evaluation Results:")
    logger.info(f"  Total Loss: {results['total_loss']:.4f}")
    logger.info(f"  MLM Accuracy: {results['mlm_accuracy']:.4f}")
    
    if 'mlm_loss' in results:
        logger.info(f"  MLM Loss: {results['mlm_loss']:.4f}")
    
    if 'perplexity' in results:
        logger.info(f"  Perplexity: {results['perplexity']:.2f}")
    
    if 'nsp_accuracy' in results:
        logger.info(f"  NSP Accuracy: {results['nsp_accuracy']:.4f}")
        logger.info(f"  NSP Loss: {results['nsp_loss']:.4f}")
    
    return results, token_level_results


def analyze_batch_predictions(batch, outputs, tokenizer):
    """Analyze predictions for a batch to understand model behavior."""
    input_ids = batch['input_ids']
    mlm_labels = batch['mlm_labels']
    mlm_logits = outputs['mlm_logits']
    
    predictions = torch.argmax(mlm_logits, dim=-1)
    
    results = []
    
    for i in range(input_ids.size(0)):  # For each example in batch
        seq_input_ids = input_ids[i]
        seq_labels = mlm_labels[i]
        seq_predictions = predictions[i]
        
        # Find masked positions
        masked_positions = (seq_labels != -100).nonzero(as_tuple=False).flatten()
        
        for pos in masked_positions:
            pos = pos.item()
            
            # Get tokens
            input_token_id = seq_input_ids[pos].item()
            true_token_id = seq_labels[pos].item()
            pred_token_id = seq_predictions[pos].item()
            
            input_token = tokenizer.id2token.get(input_token_id, '[UNK]')
            true_token = tokenizer.id2token.get(true_token_id, '[UNK]')
            pred_token = tokenizer.id2token.get(pred_token_id, '[UNK]')
            
            # Get prediction confidence (probability)
            probs = F.softmax(mlm_logits[i, pos], dim=-1)
            true_prob = probs[true_token_id].item()
            pred_prob = probs[pred_token_id].item()
            
            results.append({
                'position': pos,
                'input_token': input_token,
                'true_token': true_token,
                'predicted_token': pred_token,
                'correct': true_token_id == pred_token_id,
                'true_probability': true_prob,
                'predicted_probability': pred_prob
            })
    
    return results


def analyze_vocabulary_coverage(model, tokenizer, device, logger):
    """Analyze how well the model predicts different parts of vocabulary."""
    vocab_stats = {}
    
    # Create simple test sentences for each token
    test_sentences = []
    target_tokens = []
    
    # Sample some tokens from vocabulary (skip special tokens)
    sample_tokens = list(tokenizer.vocab.keys())[5:min(100, len(tokenizer.vocab))]
    
    for token in sample_tokens:
        # Create simple sentence with token in middle
        sentence = f"the {token} is good"
        test_sentences.append(sentence)
        target_tokens.append(token)
    
    logger.info(f"Testing vocabulary coverage on {len(test_sentences)} samples")
    
    model.eval()
    correct_predictions = 0
    
    with torch.no_grad():
        for sentence, target_token in zip(test_sentences, target_tokens):
            # Tokenize sentence
            tokens = tokenizer.tokenize(sentence)
            
            if target_token not in tokens:
                continue
                
            # Find target token position
            target_pos = tokens.index(target_token) + 1  # +1 for [CLS]
            
            # Create input with target token masked
            input_tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
            input_tokens[target_pos] = tokenizer.mask_token
            
            # Convert to IDs
            input_ids = torch.tensor([[tokenizer.vocab.get(t, tokenizer.vocab[tokenizer.unk_token]) 
                                     for t in input_tokens]], device=device)
            
            # Create attention mask
            attention_mask = torch.ones_like(input_ids)
            
            # Forward pass
            outputs = model.bert(input_ids, attention_mask=attention_mask)
            logits = model.mlm_head(outputs)
            
            # Get prediction
            pred_token_id = torch.argmax(logits[0, target_pos]).item()
            true_token_id = tokenizer.vocab.get(target_token, tokenizer.vocab[tokenizer.unk_token])
            
            if pred_token_id == true_token_id:
                correct_predictions += 1
    
    vocab_accuracy = correct_predictions / len(test_sentences)
    logger.info(f"Vocabulary coverage accuracy: {vocab_accuracy:.4f}")
    
    return vocab_accuracy


def main():
    parser = argparse.ArgumentParser(description='Evaluate BERT MLM model')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--corpus', required=True, help='Path to evaluation corpus')
    parser.add_argument('--vocab', required=True, help='Path to vocabulary file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--max_seq_length', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--output', help='Path to save evaluation results')
    parser.add_argument('--detailed', action='store_true', help='Run detailed analysis')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, config = load_model_and_config(args.checkpoint, device)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.vocab}")
    tokenizer = SimpleTokenizer(vocab_file=args.vocab)
    
    # Load corpus
    logger.info(f"Loading corpus from {args.corpus}")
    texts = load_corpus(args.corpus)
    logger.info(f"Loaded {len(texts)} texts for evaluation")
    
    # Use the max_seq_length from config to ensure consistency
    eval_max_seq_length = config.get('max_seq_length', args.max_seq_length)
    print(f"Using max_seq_length: {eval_max_seq_length} (from config: {config.get('max_seq_length', 'not set')})")
    
    # Create dataloader with config's max_seq_length
    dataloader = create_mlm_dataloader(
        texts=texts,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_length=eval_max_seq_length,
        mlm_probability=0.15,
        use_nsp=config.get('use_nsp', False),
        shuffle=False
    )
    
    # Evaluate model
    logger.info("Starting evaluation...")
    results, token_results = evaluate_model_detailed(model, dataloader, device, logger, tokenizer)
    
    # Detailed analysis if requested
    if args.detailed:
        logger.info("Running detailed analysis...")
        
        # Vocabulary coverage
        vocab_accuracy = analyze_vocabulary_coverage(model, tokenizer, device, logger)
        results['vocabulary_accuracy'] = vocab_accuracy
        
        # Analyze prediction patterns
        if token_results:
            correct_preds = [r for r in token_results if r['correct']]
            incorrect_preds = [r for r in token_results if not r['correct']]
            
            logger.info(f"\nPrediction Analysis:")
            logger.info(f"  Total predictions analyzed: {len(token_results)}")
            logger.info(f"  Correct predictions: {len(correct_preds)}")
            logger.info(f"  Incorrect predictions: {len(incorrect_preds)}")
            
            if correct_preds:
                avg_correct_prob = np.mean([r['true_probability'] for r in correct_preds])
                logger.info(f"  Average probability for correct predictions: {avg_correct_prob:.4f}")
            
            if incorrect_preds:
                avg_incorrect_prob = np.mean([r['predicted_probability'] for r in incorrect_preds])
                logger.info(f"  Average probability for incorrect predictions: {avg_incorrect_prob:.4f}")
            
            # Show some examples
            logger.info(f"\nSample correct predictions:")
            for i, result in enumerate(correct_preds[:5]):
                logger.info(f"  {result['input_token']} -> {result['true_token']} "
                           f"(prob: {result['true_probability']:.4f})")
            
            logger.info(f"\nSample incorrect predictions:")
            for i, result in enumerate(incorrect_preds[:5]):
                logger.info(f"  {result['input_token']} -> {result['predicted_token']} "
                           f"(should be: {result['true_token']}, prob: {result['predicted_probability']:.4f})")
    
    # Save results
    if args.output:
        logger.info(f"Saving results to {args.output}")
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
