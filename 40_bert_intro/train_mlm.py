"""
Training script for BERT MLM pretraining.
"""

import os
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
import logging
from tqdm import tqdm
import json
from datetime import datetime

# Local imports
from model.bert_encoder import BERTModel
from model.heads import BERTForPreTraining, compute_mlm_accuracy, compute_nsp_accuracy
from tokenization.simple_wp import SimpleTokenizer
from tokenization.dataset_mlm import create_mlm_dataloader, load_corpus


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_file=None):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
        
    return logging.getLogger(__name__)


def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']


def evaluate_model(model, dataloader, device, logger):
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0
    total_mlm_accuracy = 0
    total_nsp_accuracy = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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
            
            # Calculate metrics
            batch_size = input_ids.size(0)
            total_samples += batch_size
            total_loss += outputs['loss'].item() * batch_size
            
            # MLM accuracy
            mlm_acc = compute_mlm_accuracy(outputs['mlm_logits'], mlm_labels)
            total_mlm_accuracy += mlm_acc * batch_size
            
            # NSP accuracy (if available)
            if 'nsp_logits' in outputs:
                nsp_acc = compute_nsp_accuracy(outputs['nsp_logits'], nsp_labels)
                total_nsp_accuracy += nsp_acc * batch_size
    
    avg_loss = total_loss / total_samples
    avg_mlm_accuracy = total_mlm_accuracy / total_samples
    
    results = {
        'loss': avg_loss,
        'mlm_accuracy': avg_mlm_accuracy
    }
    
    if total_nsp_accuracy > 0:
        results['nsp_accuracy'] = total_nsp_accuracy / total_samples
    
    logger.info(f"Validation - Loss: {avg_loss:.4f}, MLM Acc: {avg_mlm_accuracy:.4f}")
    if 'nsp_accuracy' in results:
        logger.info(f"NSP Acc: {results['nsp_accuracy']:.4f}")
    
    return results


def train_epoch(model, dataloader, optimizer, scheduler, device, logger, epoch):
    """Train model for one epoch."""
    model.train()
    
    total_loss = 0
    total_mlm_loss = 0
    total_nsp_loss = 0
    total_mlm_accuracy = 0
    total_nsp_accuracy = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(progress_bar):
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
        
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        # Update metrics
        batch_size = input_ids.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size
        
        if 'mlm_loss' in outputs:
            total_mlm_loss += outputs['mlm_loss'].item() * batch_size
        
        if 'nsp_loss' in outputs:
            total_nsp_loss += outputs['nsp_loss'].item() * batch_size
        
        # Calculate accuracies
        mlm_acc = compute_mlm_accuracy(outputs['mlm_logits'], mlm_labels)
        total_mlm_accuracy += mlm_acc * batch_size
        
        if 'nsp_logits' in outputs:
            nsp_acc = compute_nsp_accuracy(outputs['nsp_logits'], nsp_labels)
            total_nsp_accuracy += nsp_acc * batch_size
        
        # Update progress bar
        avg_loss = total_loss / total_samples
        avg_mlm_acc = total_mlm_accuracy / total_samples
        
        desc = f"Epoch {epoch} - Loss: {avg_loss:.4f}, MLM Acc: {avg_mlm_acc:.4f}"
        if total_nsp_accuracy > 0:
            avg_nsp_acc = total_nsp_accuracy / total_samples
            desc += f", NSP Acc: {avg_nsp_acc:.4f}"
        
        progress_bar.set_description(desc)
        
        # Log intermediate results
        if step % 100 == 0:
            lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            logger.info(f"Step {step}: Loss={avg_loss:.4f}, MLM Acc={avg_mlm_acc:.4f}, LR={lr:.2e}")
    
    # Return average metrics for the epoch
    results = {
        'loss': total_loss / total_samples,
        'mlm_accuracy': total_mlm_accuracy / total_samples
    }
    
    if total_mlm_loss > 0:
        results['mlm_loss'] = total_mlm_loss / total_samples
    
    if total_nsp_loss > 0:
        results['nsp_loss'] = total_nsp_loss / total_samples
        results['nsp_accuracy'] = total_nsp_accuracy / total_samples
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train BERT with MLM')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--corpus', required=True, help='Path to training corpus')
    parser.add_argument('--vocab', help='Path to vocabulary file (will be created if not exists)')
    parser.add_argument('--output_dir', default='./checkpoints', help='Output directory')
    parser.add_argument('--resume', help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--eval_corpus', help='Path to evaluation corpus')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.output_dir, 'training.log')
    logger = setup_logging(log_file)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load corpus
    logger.info(f"Loading corpus from {args.corpus}")
    train_texts = load_corpus(args.corpus)
    logger.info(f"Loaded {len(train_texts)} training texts")
    
    # Load or create tokenizer
    vocab_file = args.vocab or os.path.join(args.output_dir, 'vocab.json')
    
    if os.path.exists(vocab_file):
        logger.info(f"Loading vocabulary from {vocab_file}")
        tokenizer = SimpleTokenizer(vocab_file=vocab_file)
    else:
        logger.info(f"Creating vocabulary from corpus")
        tokenizer = SimpleTokenizer(
            vocab_size=config['vocab_size'], 
            min_frequency=config.get('min_frequency', 2)
        )
        tokenizer.build_vocab(train_texts)
        tokenizer.save_vocab(vocab_file)
    
    logger.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Update config with actual vocab size
    config['vocab_size'] = tokenizer.get_vocab_size()
    
    # Create data loaders
    train_dataloader = create_mlm_dataloader(
        texts=train_texts,
        tokenizer=tokenizer,
        batch_size=config['batch_size'],
        max_seq_length=config['max_seq_length'],
        mlm_probability=config.get('mlm_probability', 0.15),
        use_nsp=config.get('use_nsp', False)
    )
    
    eval_dataloader = None
    if args.eval_corpus:
        eval_texts = load_corpus(args.eval_corpus)
        eval_dataloader = create_mlm_dataloader(
            texts=eval_texts,
            tokenizer=tokenizer,
            batch_size=config['batch_size'],
            max_seq_length=config['max_seq_length'],
            mlm_probability=config.get('mlm_probability', 0.15),
            use_nsp=config.get('use_nsp', False),
            shuffle=False
        )
    
    # Create model
    logger.info("Creating BERT model")
    bert_model = BERTModel(config)
    model = BERTForPreTraining(bert_model, config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01),
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    num_training_steps = len(train_dataloader) * config['num_epochs']
    num_warmup_steps = int(config.get('warmup_ratio', 0.05) * num_training_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, _, _ = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1
    
    # Save config
    config_save_path = os.path.join(args.output_dir, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    logger.info("Starting training")
    best_loss = float('inf')
    
    for epoch in range(start_epoch, config['num_epochs']):
        logger.info(f"Starting epoch {epoch + 1}/{config['num_epochs']}")
        
        # Training
        train_results = train_epoch(model, train_dataloader, optimizer, scheduler, 
                                  device, logger, epoch + 1)
        
        logger.info(f"Epoch {epoch + 1} - Training Loss: {train_results['loss']:.4f}, "
                   f"MLM Acc: {train_results['mlm_accuracy']:.4f}")
        
        if 'nsp_accuracy' in train_results:
            logger.info(f"NSP Acc: {train_results['nsp_accuracy']:.4f}")
        
        # Evaluation
        if eval_dataloader:
            eval_results = evaluate_model(model, eval_dataloader, device, logger)
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch + 1}.pt')
        save_checkpoint(model, optimizer, scheduler, epoch, 0, train_results['loss'], checkpoint_path)
        
        # Save best model
        current_loss = eval_results['loss'] if eval_dataloader else train_results['loss']
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_path = os.path.join(args.output_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, 0, current_loss, best_model_path)
            logger.info(f"New best model saved with loss: {best_loss:.4f}")
    
    logger.info("Training completed!")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
