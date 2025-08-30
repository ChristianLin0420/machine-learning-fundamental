"""
Dataset for Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).
Creates masked examples with attention masks and segment IDs.
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np


class MLMDataset(Dataset):
    """Dataset for Masked Language Modeling pretraining."""
    
    def __init__(self, texts: List[str], tokenizer, max_seq_length: int = 128,
                 mlm_probability: float = 0.15, nsp_probability: float = 0.5,
                 use_nsp: bool = True, short_seq_prob: float = 0.1):
        
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.nsp_probability = nsp_probability
        self.use_nsp = use_nsp
        self.short_seq_prob = short_seq_prob
        
        # Special token IDs
        self.pad_token_id = tokenizer.vocab[tokenizer.pad_token]
        self.cls_token_id = tokenizer.vocab[tokenizer.cls_token]
        self.sep_token_id = tokenizer.vocab[tokenizer.sep_token]
        self.mask_token_id = tokenizer.vocab[tokenizer.mask_token]
        
        # Process texts into sentences
        self.sentences = self._split_into_sentences(texts)
        
        # Create training examples
        self.examples = self._create_examples()
        
    def _split_into_sentences(self, texts: List[str]) -> List[str]:
        """Split texts into individual sentences."""
        sentences = []
        
        for text in texts:
            # Simple sentence splitting on periods, exclamation marks, question marks
            text_sentences = text.replace('!', '.').replace('?', '.').split('.')
            
            for sentence in text_sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Filter very short sentences
                    sentences.append(sentence)
                    
        return sentences
        
    def _create_examples(self) -> List[Dict]:
        """Create training examples with NSP pairs if enabled."""
        examples = []
        
        for i in range(len(self.sentences)):
            if self.use_nsp:
                # Create NSP example
                example = self._create_nsp_example(i)
            else:
                # Single sentence example
                tokens_a = self.tokenizer.tokenize(self.sentences[i])
                example = self._create_single_example(tokens_a)
                
            if example is not None:
                examples.append(example)
                
        return examples
        
    def _create_nsp_example(self, index: int) -> Optional[Dict]:
        """Create NSP training example with sentence pair."""
        current_sentence = self.sentences[index]
        
        # 50% chance of creating a positive (consecutive) pair
        if random.random() < self.nsp_probability:
            # Positive example - use next sentence
            if index + 1 < len(self.sentences):
                next_sentence = self.sentences[index + 1]
                is_next = 1  # True
            else:
                # If no next sentence, use random sentence
                next_sentence = random.choice(self.sentences)
                is_next = 0  # False
        else:
            # Negative example - use random sentence
            next_sentence = random.choice(self.sentences)
            is_next = 0  # False
            
        tokens_a = self.tokenizer.tokenize(current_sentence)
        tokens_b = self.tokenizer.tokenize(next_sentence)
        
        return self._create_pair_example(tokens_a, tokens_b, is_next)
        
    def _create_pair_example(self, tokens_a: List[str], tokens_b: List[str], 
                           is_next: int) -> Optional[Dict]:
        """Create example from sentence pair."""
        
        # Truncate sequences to fit in max_seq_length
        # Account for [CLS], [SEP], [SEP] tokens
        max_tokens_for_doc = self.max_seq_length - 3
        
        # More aggressive truncation to ensure we stay within limits
        if len(tokens_a) + len(tokens_b) > max_tokens_for_doc:
            # Truncate proportionally
            total_len = len(tokens_a) + len(tokens_b)
            ratio_a = len(tokens_a) / total_len
            max_a = int(max_tokens_for_doc * ratio_a)
            max_b = max_tokens_for_doc - max_a
            
            tokens_a = tokens_a[:max_a]
            tokens_b = tokens_b[:max_b]
                
        # Create sequence: [CLS] tokens_a [SEP] tokens_b [SEP]
        tokens = [self.tokenizer.cls_token] + tokens_a + [self.tokenizer.sep_token]
        segment_ids = [0] * len(tokens)
        
        tokens += tokens_b + [self.tokenizer.sep_token]
        segment_ids += [1] * (len(tokens_b) + 1)
        
        # Final truncation to ensure we don't exceed max_seq_length
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
            segment_ids = segment_ids[:self.max_seq_length]
        
        # Convert to IDs
        input_ids = [self.tokenizer.vocab.get(token, self.tokenizer.vocab[self.tokenizer.unk_token]) 
                    for token in tokens]
        
        return {
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'is_next': is_next
        }
        
    def _create_single_example(self, tokens: List[str]) -> Optional[Dict]:
        """Create example from single sentence (no NSP)."""
        
        # Truncate if necessary
        max_tokens = self.max_seq_length - 2  # Account for [CLS] and [SEP]
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            
        # Add special tokens
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        
        # Ensure we don't exceed max_seq_length
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]
        
        # Convert to IDs
        input_ids = [self.tokenizer.vocab.get(token, self.tokenizer.vocab[self.tokenizer.unk_token]) 
                    for token in tokens]
        
        # All tokens have segment_id = 0 for single sentence
        segment_ids = [0] * len(input_ids)
        
        return {
            'input_ids': input_ids,
            'segment_ids': segment_ids,
            'is_next': None
        }
        
    def _apply_mlm_masking(self, input_ids: List[int], segment_ids: List[int]) -> Tuple[List[int], List[int]]:
        """Apply MLM masking to input sequence."""
        
        masked_input_ids = input_ids.copy()
        mlm_labels = [-100] * len(input_ids)  # -100 will be ignored in loss
        
        # Don't mask special tokens
        special_token_ids = {self.cls_token_id, self.sep_token_id, self.pad_token_id}
        
        # Select tokens to mask (15% probability)
        for i, token_id in enumerate(input_ids):
            if token_id in special_token_ids:
                continue
                
            if random.random() < self.mlm_probability:
                mlm_labels[i] = token_id  # Store original token for loss
                
                # Masking strategy: 80% [MASK], 10% random, 10% unchanged
                rand = random.random()
                if rand < 0.8:
                    # 80% of time: replace with [MASK]
                    masked_input_ids[i] = self.mask_token_id
                elif rand < 0.9:
                    # 10% of time: replace with random token
                    masked_input_ids[i] = random.randint(5, len(self.tokenizer.vocab) - 1)
                # 10% of time: keep unchanged (no else needed)
                
        return masked_input_ids, mlm_labels
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict:
        example = self.examples[idx]
        
        # Ensure input sequence doesn't exceed max length BEFORE any processing
        input_ids = example['input_ids'][:self.max_seq_length]
        segment_ids = example['segment_ids'][:self.max_seq_length]
        
        # Apply MLM masking
        masked_input_ids, mlm_labels = self._apply_mlm_masking(input_ids, segment_ids)
        
        # Ensure everything is exactly max_seq_length
        seq_len = len(masked_input_ids)
        
        if seq_len > self.max_seq_length:
            # Force truncate everything to max_seq_length
            masked_input_ids = masked_input_ids[:self.max_seq_length]
            segment_ids = segment_ids[:self.max_seq_length]
            mlm_labels = mlm_labels[:self.max_seq_length]
        elif seq_len < self.max_seq_length:
            # Pad to exact length
            padding_length = self.max_seq_length - seq_len
            masked_input_ids.extend([self.pad_token_id] * padding_length)
            segment_ids.extend([0] * padding_length)
            mlm_labels.extend([-100] * padding_length)
        
        # Double check lengths are exactly correct
        assert len(masked_input_ids) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(mlm_labels) == self.max_seq_length
            
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if token_id != self.pad_token_id else 0 
                         for token_id in masked_input_ids]
        
        result = {
            'input_ids': torch.tensor(masked_input_ids, dtype=torch.long),
            'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'mlm_labels': torch.tensor(mlm_labels, dtype=torch.long)
        }
        
        # Add NSP label if available
        if example['is_next'] is not None:
            result['nsp_labels'] = torch.tensor(example['is_next'], dtype=torch.long)
            
        return result


def custom_collate_fn(batch):
    """Custom collate function to ensure consistent tensor sizes."""
    collated = {}
    
    for key in batch[0].keys():
        if key in ['input_ids', 'segment_ids', 'attention_mask', 'mlm_labels']:
            # Stack tensors for sequence-level keys
            tensors = [item[key] for item in batch]
            
            # Ensure all tensors have the same size
            max_len = max(tensor.size(0) for tensor in tensors)
            
            # Pad if necessary
            padded_tensors = []
            for tensor in tensors:
                if tensor.size(0) < max_len:
                    if key in ['input_ids', 'segment_ids', 'attention_mask']:
                        pad_value = 0
                    else:  # mlm_labels
                        pad_value = -100
                    
                    padding = torch.full((max_len - tensor.size(0),), pad_value, dtype=tensor.dtype)
                    tensor = torch.cat([tensor, padding])
                elif tensor.size(0) > max_len:
                    tensor = tensor[:max_len]
                    
                padded_tensors.append(tensor)
            
            collated[key] = torch.stack(padded_tensors)
        else:
            # For scalar values like nsp_labels
            collated[key] = torch.tensor([item[key] for item in batch if key in item])
            
    return collated


def create_mlm_dataloader(texts: List[str], tokenizer, batch_size: int = 32,
                         max_seq_length: int = 128, mlm_probability: float = 0.15,
                         use_nsp: bool = True, shuffle: bool = True) -> DataLoader:
    """Create DataLoader for MLM pretraining."""
    
    dataset = MLMDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        mlm_probability=mlm_probability,
        use_nsp=use_nsp
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True,
        collate_fn=custom_collate_fn
    )


def load_corpus(corpus_file: str) -> List[str]:
    """Load text corpus from file."""
    with open(corpus_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Filter empty lines
    texts = [line.strip() for line in lines if line.strip()]
    
    return texts


if __name__ == "__main__":
    # Example usage
    from tokenization.simple_wp import SimpleTokenizer
    
    # Sample texts
    texts = [
        "The quick brown fox jumps over the lazy dog. This is a sample sentence.",
        "BERT is a transformer model. It uses masked language modeling for pretraining.",
        "Natural language processing is fascinating. We can teach computers to understand text.",
        "Machine learning models require large datasets. Training can take a long time.",
        "Deep learning has revolutionized AI. Neural networks are very powerful."
    ]
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.build_vocab(texts)
    
    # Create dataset
    dataset = MLMDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_seq_length=64,
        use_nsp=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Sample a batch
    sample = dataset[0]
    print(f"\nSample batch:")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Segment IDs shape: {sample['segment_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"MLM labels shape: {sample['mlm_labels'].shape}")
    
    if 'nsp_labels' in sample:
        print(f"NSP label: {sample['nsp_labels']}")
        
    # Decode example
    print(f"\nDecoded input: {tokenizer.decode(sample['input_ids'].tolist())}")
    
    # Show masked positions
    masked_positions = (sample['mlm_labels'] != -100).nonzero(as_tuple=False).flatten()
    print(f"Masked positions: {masked_positions.tolist()}")
    
    for pos in masked_positions:
        original_token_id = sample['mlm_labels'][pos].item()
        masked_token_id = sample['input_ids'][pos].item()
        original_token = tokenizer.id2token[original_token_id]
        masked_token = tokenizer.id2token[masked_token_id]
        print(f"Position {pos}: {original_token} -> {masked_token}")
