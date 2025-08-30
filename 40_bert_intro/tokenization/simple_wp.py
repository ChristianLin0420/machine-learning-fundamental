"""
Simple WordPiece-like tokenizer for BERT.
For simplicity, this implements whitespace tokenization with subword splitting.
"""

import re
import json
from collections import Counter, OrderedDict
from typing import List, Dict, Tuple


class SimpleTokenizer:
    """Simple tokenizer with special tokens and basic subword support."""
    
    def __init__(self, vocab_file=None, vocab_size=5000, min_frequency=2):
        # Special tokens
        self.special_tokens = {
            '[PAD]': 0,
            '[CLS]': 1, 
            '[SEP]': 2,
            '[MASK]': 3,
            '[UNK]': 4
        }
        
        self.pad_token = '[PAD]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.mask_token = '[MASK]'
        self.unk_token = '[UNK]'
        
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        if vocab_file:
            self.load_vocab(vocab_file)
        else:
            self.vocab = self.special_tokens.copy()
            self.id2token = {v: k for k, v in self.vocab.items()}
            
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from text corpus."""
        # Tokenize all texts and count frequencies
        all_tokens = []
        for text in texts:
            tokens = self._basic_tokenize(text)
            all_tokens.extend(tokens)
            
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Build vocabulary starting with special tokens
        self.vocab = self.special_tokens.copy()
        
        # Add most frequent tokens up to vocab_size
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        
        for token, count in sorted_tokens:
            if len(self.vocab) >= self.vocab_size:
                break
            if count >= self.min_frequency and token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                
        # Create reverse mapping
        self.id2token = {v: k for k, v in self.vocab.items()}
        
        print(f"Built vocabulary with {len(self.vocab)} tokens")
        
    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic tokenization: lowercase, split on whitespace and punctuation."""
        # Lowercase and clean
        text = text.lower().strip()
        
        # Split on whitespace and basic punctuation
        tokens = re.findall(r'\w+|[^\w\s]', text)
        
        return tokens
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens."""
        tokens = self._basic_tokenize(text)
        
        # Handle unknown tokens
        result = []
        for token in tokens:
            if token in self.vocab:
                result.append(token)
            else:
                # Simple subword splitting for OOV tokens
                subwords = self._split_into_subwords(token)
                result.extend(subwords)
                
        return result
        
    def _split_into_subwords(self, token: str) -> List[str]:
        """Split unknown token into subwords or return [UNK]."""
        # Very simple subword splitting - try prefixes and suffixes
        if len(token) <= 3:
            return [self.unk_token]
            
        # Try to find known subwords
        for i in range(1, len(token)):
            prefix = token[:i]
            suffix = token[i:]
            
            if prefix in self.vocab and suffix in self.vocab:
                return [prefix, suffix]
                
        # If no split found, return [UNK]
        return [self.unk_token]
        
    def encode(self, text: str, add_special_tokens=True) -> List[int]:
        """Encode text to token IDs."""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]
            
        # Convert to IDs
        token_ids = []
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.vocab[self.unk_token]))
            
        return token_ids
        
    def decode(self, token_ids: List[int], skip_special_tokens=True) -> str:
        """Decode token IDs back to text."""
        tokens = []
        for token_id in token_ids:
            token = self.id2token.get(token_id, self.unk_token)
            
            if skip_special_tokens and token in self.special_tokens:
                continue
                
            tokens.append(token)
            
        return ' '.join(tokens)
        
    def encode_pair(self, text_a: str, text_b: str) -> Tuple[List[int], List[int]]:
        """
        Encode a pair of texts for NSP task.
        
        Returns:
            input_ids: [CLS] text_a [SEP] text_b [SEP]
            segment_ids: [0, 0, ..., 1, 1, ...]
        """
        tokens_a = self.tokenize(text_a)
        tokens_b = self.tokenize(text_b)
        
        # Create input sequence: [CLS] text_a [SEP] text_b [SEP]
        tokens = [self.cls_token] + tokens_a + [self.sep_token] + tokens_b + [self.sep_token]
        
        # Create segment IDs
        segment_ids = ([0] * (len(tokens_a) + 2) +  # [CLS] text_a [SEP]
                      [1] * (len(tokens_b) + 1))     # text_b [SEP]
        
        # Convert to IDs
        input_ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
        
        return input_ids, segment_ids
        
    def save_vocab(self, vocab_file: str) -> None:
        """Save vocabulary to file."""
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f, indent=2)
            
    def load_vocab(self, vocab_file: str) -> None:
        """Load vocabulary from file."""
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f)
        self.id2token = {v: k for k, v in self.vocab.items()}
        
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
        
    def pad_sequences(self, sequences: List[List[int]], max_length: int, 
                     pad_token_id: int = None) -> List[List[int]]:
        """Pad sequences to the same length."""
        if pad_token_id is None:
            pad_token_id = self.vocab[self.pad_token]
            
        padded = []
        for seq in sequences:
            if len(seq) >= max_length:
                padded.append(seq[:max_length])
            else:
                padding = [pad_token_id] * (max_length - len(seq))
                padded.append(seq + padding)
                
        return padded


def create_vocab_from_corpus(corpus_file: str, vocab_file: str, 
                           vocab_size: int = 5000, min_frequency: int = 2):
    """Create vocabulary from a text corpus file."""
    
    # Read corpus
    with open(corpus_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()
        
    # Create tokenizer and build vocab
    tokenizer = SimpleTokenizer(vocab_size=vocab_size, min_frequency=min_frequency)
    tokenizer.build_vocab(texts)
    
    # Save vocabulary
    tokenizer.save_vocab(vocab_file)
    
    print(f"Vocabulary saved to {vocab_file}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    
    return tokenizer


if __name__ == "__main__":
    # Example usage
    texts = [
        "Hello world! This is a simple tokenizer.",
        "BERT uses WordPiece tokenization for subword units.",
        "We implement a simplified version for learning purposes."
    ]
    
    tokenizer = SimpleTokenizer(vocab_size=100)
    tokenizer.build_vocab(texts)
    
    # Test tokenization
    text = "Hello BERT tokenizer!"
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)
    decoded = tokenizer.decode(token_ids)
    
    print(f"Original: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token IDs: {token_ids}")
    print(f"Decoded: {decoded}")
    
    # Test pair encoding
    text_a = "This is the first sentence."
    text_b = "This is the second sentence."
    input_ids, segment_ids = tokenizer.encode_pair(text_a, text_b)
    
    print(f"\nPair encoding:")
    print(f"Text A: {text_a}")
    print(f"Text B: {text_b}")
    print(f"Input IDs: {input_ids}")
    print(f"Segment IDs: {segment_ids}")

