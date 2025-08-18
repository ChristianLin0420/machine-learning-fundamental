#!/usr/bin/env python3
"""
Data preprocessing utilities for sequence classification tasks.

This module provides comprehensive text preprocessing, tokenization, 
and dataset creation utilities for sequence classification.
"""

import numpy as np
import pandas as pd
import re
import string
from typing import List, Tuple, Dict, Optional, Union
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
import requests
import os
import zipfile
import tarfile
from urllib.parse import urlparse

class TextPreprocessor:
    """Text preprocessing utility with various cleaning options."""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 remove_digits: bool = False,
                 remove_stopwords: bool = False,
                 min_word_length: int = 1,
                 max_word_length: int = 50):
        """
        Initialize text preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_digits: Remove numeric digits
            remove_stopwords: Remove common stop words
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length
        self.max_word_length = max_word_length
        
        # Common English stop words
        self.stopwords = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
            'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
            'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
            'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
            'with', 'through', 'during', 'before', 'after', 'above', 'below', 
            'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once'
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean a single text string according to preprocessor settings.
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove digits if requested
        if self.remove_digits:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation if requested
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        text = self.clean_text(text)
        tokens = text.split()
        
        # Filter by word length
        tokens = [
            token for token in tokens 
            if self.min_word_length <= len(token) <= self.max_word_length
        ]
        
        # Remove stopwords if requested
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        return tokens
    
    def preprocess_texts(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of tokenized texts
        """
        return [self.tokenize(text) for text in texts]


class Vocabulary:
    """Vocabulary management for text data."""
    
    def __init__(self, 
                 max_vocab_size: Optional[int] = None,
                 min_freq: int = 1,
                 special_tokens: Optional[List[str]] = None):
        """
        Initialize vocabulary.
        
        Args:
            max_vocab_size: Maximum vocabulary size
            min_freq: Minimum token frequency to include
            special_tokens: Special tokens to include
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        
        # Define special tokens
        if special_tokens is None:
            special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        self.special_tokens = special_tokens
        
        # Initialize mappings
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.token_freq = Counter()
        self.vocab_size = 0
        
        # Add special tokens
        for token in self.special_tokens:
            self._add_token(token)
    
    def _add_token(self, token: str) -> int:
        """Add a token to vocabulary."""
        if token not in self.token_to_idx:
            idx = len(self.token_to_idx)
            self.token_to_idx[token] = idx
            self.idx_to_token[idx] = token
            self.vocab_size += 1
        return self.token_to_idx[token]
    
    def build_vocab(self, tokenized_texts: List[List[str]]) -> None:
        """
        Build vocabulary from tokenized texts.
        
        Args:
            tokenized_texts: List of tokenized text sequences
        """
        # Count token frequencies
        for tokens in tokenized_texts:
            self.token_freq.update(tokens)
        
        # Sort tokens by frequency
        sorted_tokens = sorted(
            self.token_freq.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Add tokens to vocabulary based on frequency and limits
        for token, freq in sorted_tokens:
            if freq < self.min_freq:
                break
            
            if (self.max_vocab_size is not None and 
                len(self.token_to_idx) >= self.max_vocab_size):
                break
            
            if token not in self.token_to_idx:
                self._add_token(token)
    
    def encode(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to indices.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of token indices
        """
        unk_idx = self.token_to_idx.get('<UNK>', 1)
        return [self.token_to_idx.get(token, unk_idx) for token in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        """
        Convert indices to tokens.
        
        Args:
            indices: List of token indices
            
        Returns:
            List of tokens
        """
        return [self.idx_to_token.get(idx, '<UNK>') for idx in indices]
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    def get_padding_idx(self) -> int:
        """Get padding token index."""
        return self.token_to_idx.get('<PAD>', 0)


class SequenceDataset:
    """Dataset class for sequence classification."""
    
    def __init__(self, 
                 texts: List[str], 
                 labels: List[int],
                 preprocessor: TextPreprocessor,
                 vocabulary: Vocabulary,
                 max_length: Optional[int] = None,
                 padding: str = 'post',
                 truncating: str = 'post'):
        """
        Initialize sequence dataset.
        
        Args:
            texts: List of text strings
            labels: List of labels
            preprocessor: Text preprocessor
            vocabulary: Vocabulary object
            max_length: Maximum sequence length
            padding: Padding strategy ('pre' or 'post')
            truncating: Truncating strategy ('pre' or 'post')
        """
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.vocabulary = vocabulary
        self.max_length = max_length
        self.padding = padding
        self.truncating = truncating
        
        # Preprocess and encode texts
        self.tokenized_texts = self.preprocessor.preprocess_texts(texts)
        self.encoded_sequences = [
            self.vocabulary.encode(tokens) for tokens in self.tokenized_texts
        ]
        
        # Determine max length if not provided
        if self.max_length is None:
            self.max_length = max(len(seq) for seq in self.encoded_sequences)
        
        # Pad/truncate sequences
        self.padded_sequences = self._pad_sequences()
        
        # Convert to numpy arrays
        self.sequences = np.array(self.padded_sequences)
        self.labels = np.array(labels)
    
    def _pad_sequences(self) -> List[List[int]]:
        """Pad sequences to uniform length."""
        padded = []
        pad_idx = self.vocabulary.get_padding_idx()
        
        for seq in self.encoded_sequences:
            # Truncate if too long
            if len(seq) > self.max_length:
                if self.truncating == 'pre':
                    seq = seq[-self.max_length:]
                else:  # 'post'
                    seq = seq[:self.max_length]
            
            # Pad if too short
            elif len(seq) < self.max_length:
                pad_length = self.max_length - len(seq)
                if self.padding == 'pre':
                    seq = [pad_idx] * pad_length + seq
                else:  # 'post'
                    seq = seq + [pad_idx] * pad_length
            
            padded.append(seq)
        
        return padded
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get a single sample."""
        return self.sequences[idx], self.labels[idx]
    
    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Get a batch of samples."""
        return self.sequences[indices], self.labels[indices]


class DatasetLoader:
    """Utility class for loading common sequence classification datasets."""
    
    @staticmethod
    def load_imdb_sample(n_samples: int = 1000) -> Tuple[List[str], List[int]]:
        """
        Load a sample of IMDB movie reviews for sentiment analysis.
        
        Args:
            n_samples: Number of samples to load
            
        Returns:
            Tuple of (texts, labels) where labels are 0=negative, 1=positive
        """
        # Create synthetic IMDB-like data for demonstration
        positive_phrases = [
            "excellent movie", "great acting", "wonderful story", "amazing film",
            "brilliant performance", "outstanding cinematography", "superb direction",
            "incredible plot", "fantastic characters", "marvelous experience",
            "beautiful visuals", "compelling narrative", "exceptional quality",
            "remarkable achievement", "stunning performance", "masterful storytelling",
            "captivating story", "breathtaking scenes", "phenomenal acting",
            "unforgettable movie"
        ]
        
        negative_phrases = [
            "terrible movie", "poor acting", "boring story", "awful film",
            "weak performance", "bad cinematography", "poor direction",
            "confusing plot", "flat characters", "disappointing experience",
            "ugly visuals", "weak narrative", "low quality",
            "poor achievement", "mediocre performance", "bad storytelling",
            "uninteresting story", "dull scenes", "overacting",
            "forgettable movie"
        ]
        
        texts = []
        labels = []
        
        # Generate positive samples
        for i in range(n_samples // 2):
            phrase = np.random.choice(positive_phrases)
            text = f"This {phrase} really impressed me. The movie was {np.random.choice(['great', 'excellent', 'wonderful', 'amazing'])}."
            texts.append(text)
            labels.append(1)
        
        # Generate negative samples
        for i in range(n_samples // 2):
            phrase = np.random.choice(negative_phrases)
            text = f"This {phrase} really disappointed me. The movie was {np.random.choice(['bad', 'terrible', 'awful', 'poor'])}."
            texts.append(text)
            labels.append(0)
        
        # Shuffle
        combined = list(zip(texts, labels))
        np.random.shuffle(combined)
        texts, labels = zip(*combined)
        
        return list(texts), list(labels)
    
    @staticmethod
    def load_20newsgroups_sample(categories: Optional[List[str]] = None,
                                n_samples: int = 1000) -> Tuple[List[str], List[int], List[str]]:
        """
        Load a sample of 20 newsgroups dataset.
        
        Args:
            categories: List of categories to include
            n_samples: Number of samples to load
            
        Returns:
            Tuple of (texts, labels, category_names)
        """
        if categories is None:
            categories = ['alt.atheism', 'soc.religion.christian', 
                         'comp.graphics', 'sci.med']
        
        try:
            # Try to load real 20newsgroups data
            newsgroups = fetch_20newsgroups(
                subset='train',
                categories=categories,
                remove=('headers', 'footers', 'quotes')
            )
            
            # Sample if we have more than requested
            if len(newsgroups.data) > n_samples:
                indices = np.random.choice(
                    len(newsgroups.data), 
                    n_samples, 
                    replace=False
                )
                texts = [newsgroups.data[i] for i in indices]
                labels = [newsgroups.target[i] for i in indices]
            else:
                texts = newsgroups.data
                labels = newsgroups.target.tolist()
            
            return texts, labels, newsgroups.target_names
            
        except Exception as e:
            print(f"Failed to load 20newsgroups: {e}")
            print("Generating synthetic newsgroup-like data...")
            
            # Generate synthetic data
            category_texts = {
                'tech': ['computer', 'software', 'programming', 'algorithm', 'data'],
                'sports': ['football', 'basketball', 'soccer', 'tennis', 'game'],
                'politics': ['government', 'election', 'policy', 'vote', 'democracy'],
                'science': ['research', 'experiment', 'discovery', 'theory', 'analysis']
            }
            
            texts = []
            labels = []
            category_names = list(category_texts.keys())
            
            for i in range(n_samples):
                category_idx = i % len(category_names)
                category = category_names[category_idx]
                words = np.random.choice(category_texts[category], 5)
                text = f"This is a discussion about {' and '.join(words)}. " + \
                       f"The topic of {category} is very interesting and important."
                texts.append(text)
                labels.append(category_idx)
            
            return texts, labels, category_names
    
    @staticmethod
    def create_synthetic_sequences(n_samples: int = 1000,
                                  seq_length_range: Tuple[int, int] = (10, 50),
                                  vocab_size: int = 1000,
                                  n_classes: int = 3) -> Tuple[List[str], List[int]]:
        """
        Create synthetic sequence classification data.
        
        Args:
            n_samples: Number of samples to generate
            seq_length_range: Range of sequence lengths
            vocab_size: Vocabulary size
            n_classes: Number of classes
            
        Returns:
            Tuple of (texts, labels)
        """
        # Create vocabulary
        vocab = [f"word_{i}" for i in range(vocab_size)]
        
        texts = []
        labels = []
        
        for i in range(n_samples):
            # Random sequence length
            seq_length = np.random.randint(seq_length_range[0], seq_length_range[1])
            
            # Generate sequence with class-specific patterns
            label = i % n_classes
            
            # Create class-specific word preferences
            if label == 0:
                # Class 0: prefer words 0-333
                word_probs = np.ones(vocab_size)
                word_probs[:vocab_size//3] *= 3
            elif label == 1:
                # Class 1: prefer words 333-666
                word_probs = np.ones(vocab_size)
                word_probs[vocab_size//3:2*vocab_size//3] *= 3
            else:
                # Class 2: prefer words 666-999
                word_probs = np.ones(vocab_size)
                word_probs[2*vocab_size//3:] *= 3
            
            word_probs = word_probs / word_probs.sum()
            
            # Sample words
            words = np.random.choice(vocab, seq_length, p=word_probs)
            text = ' '.join(words)
            
            texts.append(text)
            labels.append(label)
        
        return texts, labels


def create_sequence_classification_data(dataset_name: str = "imdb",
                                      n_samples: int = 1000,
                                      test_size: float = 0.2,
                                      val_size: float = 0.1,
                                      max_vocab_size: int = 10000,
                                      max_length: int = 100,
                                      random_state: int = 42) -> Dict:
    """
    Create processed sequence classification datasets.
    
    Args:
        dataset_name: Name of dataset to load
        n_samples: Number of samples
        test_size: Test set proportion
        val_size: Validation set proportion
        max_vocab_size: Maximum vocabulary size
        max_length: Maximum sequence length
        random_state: Random seed
        
    Returns:
        Dictionary containing processed datasets and metadata
    """
    np.random.seed(random_state)
    
    # Load raw data
    loader = DatasetLoader()
    
    if dataset_name.lower() == "imdb":
        texts, labels = loader.load_imdb_sample(n_samples)
        class_names = ['negative', 'positive']
    elif dataset_name.lower() == "20newsgroups":
        texts, labels, class_names = loader.load_20newsgroups_sample(n_samples=n_samples)
    elif dataset_name.lower() == "synthetic":
        texts, labels = loader.create_synthetic_sequences(n_samples)
        class_names = ['class_0', 'class_1', 'class_2']
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    if val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=random_state, stratify=y_temp
        )
    else:
        X_train, X_val, y_train, y_val = X_temp, [], y_temp, []
    
    # Create preprocessor and vocabulary
    preprocessor = TextPreprocessor(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=False
    )
    
    # Build vocabulary on training data
    train_tokens = preprocessor.preprocess_texts(X_train)
    vocabulary = Vocabulary(max_vocab_size=max_vocab_size, min_freq=2)
    vocabulary.build_vocab(train_tokens)
    
    # Create datasets
    train_dataset = SequenceDataset(
        X_train, y_train, preprocessor, vocabulary, max_length
    )
    
    test_dataset = SequenceDataset(
        X_test, y_test, preprocessor, vocabulary, max_length
    )
    
    val_dataset = None
    if val_size > 0:
        val_dataset = SequenceDataset(
            X_val, y_val, preprocessor, vocabulary, max_length
        )
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'vocabulary': vocabulary,
        'preprocessor': preprocessor,
        'class_names': class_names,
        'n_classes': len(class_names),
        'vocab_size': len(vocabulary),
        'max_length': max_length,
        'dataset_info': {
            'name': dataset_name,
            'n_samples': n_samples,
            'train_size': len(X_train),
            'val_size': len(X_val) if val_size > 0 else 0,
            'test_size': len(X_test)
        }
    }


if __name__ == "__main__":
    # Test the data preprocessing utilities
    print("Testing Sequence Classification Data Preprocessing")
    print("=" * 60)
    
    # Test different datasets
    datasets = ["imdb", "20newsgroups", "synthetic"]
    
    for dataset_name in datasets:
        print(f"\nTesting {dataset_name} dataset:")
        print("-" * 40)
        
        try:
            data = create_sequence_classification_data(
                dataset_name=dataset_name,
                n_samples=200,
                max_vocab_size=1000,
                max_length=50
            )
            
            print(f"Dataset: {data['dataset_info']['name']}")
            print(f"Classes: {data['class_names']}")
            print(f"Vocabulary size: {data['vocab_size']}")
            print(f"Max sequence length: {data['max_length']}")
            print(f"Train samples: {data['dataset_info']['train_size']}")
            print(f"Test samples: {data['dataset_info']['test_size']}")
            
            # Show sample
            train_dataset = data['train_dataset']
            sample_seq, sample_label = train_dataset[0]
            sample_tokens = data['vocabulary'].decode(sample_seq.tolist())
            print(f"Sample sequence: {' '.join(sample_tokens[:10])}...")
            print(f"Sample label: {sample_label} ({data['class_names'][sample_label]})")
            
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
    
    print("\nData preprocessing utilities test completed!")