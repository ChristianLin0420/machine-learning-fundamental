"""
Task-specific heads for BERT pretraining and fine-tuning.
Includes MLM (Masked Language Modeling) and NSP (Next Sentence Prediction) heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLMHead(nn.Module):
    """Masked Language Modeling head for BERT pretraining."""
    
    def __init__(self, d_model, vocab_size, layer_norm_eps=1e-12, 
                 weight_tie=True, input_embeddings=None):
        super().__init__()
        
        self.transform = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.activation = nn.GELU()
        
        if weight_tie and input_embeddings is not None:
            # Weight tying: share parameters with input embeddings
            self.decoder = nn.Linear(d_model, vocab_size, bias=False)
            self.decoder.weight = input_embeddings.weight
            self.bias = nn.Parameter(torch.zeros(vocab_size))
        else:
            self.decoder = nn.Linear(d_model, vocab_size)
            self.bias = None
            
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # Transform hidden states
        hidden_states = self.transform(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        
        # Project to vocabulary
        logits = self.decoder(hidden_states)
        if self.bias is not None:
            logits += self.bias
            
        return logits


class NSPHead(nn.Module):
    """Next Sentence Prediction head for BERT pretraining."""
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 2)  # Binary classification
        
    def forward(self, pooled_output):
        """
        Args:
            pooled_output: (batch_size, d_model) - [CLS] token representation
        
        Returns:
            logits: (batch_size, 2) - [not_next, is_next]
        """
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class SOPHead(nn.Module):
    """Sentence Order Prediction head (alternative to NSP)."""
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, 2)  # Binary classification
        
    def forward(self, pooled_output):
        """
        Args:
            pooled_output: (batch_size, d_model) - [CLS] token representation
        
        Returns:
            logits: (batch_size, 2) - [correct_order, swapped_order]
        """
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class BERTPooler(nn.Module):
    """Pooler to get [CLS] token representation for NSP/SOP tasks."""
    
    def __init__(self, d_model):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
        
        Returns:
            pooled_output: (batch_size, d_model) - [CLS] token representation
        """
        # Take [CLS] token (first token)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BERTForPreTraining(nn.Module):
    """BERT model for pretraining with MLM and NSP/SOP heads."""
    
    def __init__(self, bert_model, config):
        super().__init__()
        
        self.bert = bert_model
        self.config = config
        
        # MLM head
        input_embeddings = self.bert.get_input_embeddings()
        self.mlm_head = MLMHead(
            d_model=config['d_model'],
            vocab_size=config['vocab_size'],
            weight_tie=config.get('weight_tie_mlm', True),
            input_embeddings=input_embeddings
        )
        
        # Pooler and NSP/SOP head (optional)
        if config.get('use_nsp', False) or config.get('use_sop', False):
            self.pooler = BERTPooler(config['d_model'])
            
            if config.get('use_nsp', False):
                self.nsp_head = NSPHead(config['d_model'])
            elif config.get('use_sop', False):
                self.sop_head = SOPHead(config['d_model'])
        else:
            self.pooler = None
            self.nsp_head = None
            self.sop_head = None
            
    def forward(self, input_ids, segment_ids=None, attention_mask=None, 
                masked_lm_labels=None, next_sentence_labels=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            segment_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            masked_lm_labels: (batch_size, seq_len) - labels for MLM, -100 for non-masked
            next_sentence_labels: (batch_size,) - labels for NSP/SOP
        
        Returns:
            dict with 'mlm_logits', 'nsp_logits' (if applicable), 'loss'
        """
        # Get hidden states from BERT
        hidden_states = self.bert(input_ids, segment_ids, attention_mask)
        
        outputs = {}
        total_loss = 0
        
        # MLM loss
        mlm_logits = self.mlm_head(hidden_states)
        outputs['mlm_logits'] = mlm_logits
        
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index is ignored
            mlm_loss = loss_fct(mlm_logits.view(-1, self.config['vocab_size']), 
                              masked_lm_labels.view(-1))
            total_loss += mlm_loss
            outputs['mlm_loss'] = mlm_loss
            
        # NSP/SOP loss
        if self.pooler is not None:
            pooled_output = self.pooler(hidden_states)
            
            if self.nsp_head is not None:
                nsp_logits = self.nsp_head(pooled_output)
                outputs['nsp_logits'] = nsp_logits
                
                if next_sentence_labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    nsp_loss = loss_fct(nsp_logits, next_sentence_labels)
                    total_loss += nsp_loss
                    outputs['nsp_loss'] = nsp_loss
                    
            elif self.sop_head is not None:
                sop_logits = self.sop_head(pooled_output)
                outputs['sop_logits'] = sop_logits
                
                if next_sentence_labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    sop_loss = loss_fct(sop_logits, next_sentence_labels)
                    total_loss += sop_loss
                    outputs['sop_loss'] = sop_loss
                    
        if total_loss > 0:
            outputs['loss'] = total_loss
            
        return outputs


def compute_mlm_accuracy(logits, labels):
    """
    Compute masked language modeling accuracy.
    
    Args:
        logits: (batch_size, seq_len, vocab_size)
        labels: (batch_size, seq_len) - labels with -100 for non-masked tokens
    
    Returns:
        accuracy: float
    """
    predictions = torch.argmax(logits, dim=-1)
    
    # Only consider positions where labels != -100 (masked positions)
    mask = (labels != -100)
    if mask.sum() == 0:
        return 0.0
        
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()


def compute_nsp_accuracy(logits, labels):
    """
    Compute next sentence prediction accuracy.
    
    Args:
        logits: (batch_size, 2)
        labels: (batch_size,)
    
    Returns:
        accuracy: float
    """
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).sum()
    accuracy = correct.float() / labels.size(0)
    return accuracy.item()

