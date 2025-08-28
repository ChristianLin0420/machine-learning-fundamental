"""
Decoding script for Transformer inference.
Supports greedy decoding and beam search for sequence generation.
"""

import torch
import torch.nn.functional as F
import argparse
import random
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
from heapq import heappush, heappop

from model import Transformer
from utils import (
    ToyDataset,
    create_padding_mask,
    create_combined_mask,
    compute_sequence_accuracy,
    load_checkpoint
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class BeamSearchDecoder:
    """
    Beam search decoder for sequence generation.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        beam_size: int = 5,
        max_len: int = 64,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0
    ):
        """
        Initialize beam search decoder.
        
        Args:
            model: Transformer model
            beam_size: Number of beams
            max_len: Maximum generation length
            sos_token_id: Start-of-sequence token ID
            eos_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID
        """
        self.model = model
        self.beam_size = beam_size
        self.max_len = max_len
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        
    def decode(
        self, 
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Decode using beam search.
        
        Args:
            src: Source sequence of shape (1, src_len)
            src_mask: Source padding mask
            
        Returns:
            best_sequence: Best generated sequence
            best_score: Score of best sequence
        """
        device = src.device
        batch_size = src.size(0)
        assert batch_size == 1, "Beam search only supports batch size 1"
        
        # Encode source
        with torch.no_grad():
            encoder_output, _ = self.model.encode(src, src_mask)
        
        # Initialize beams: (score, sequence)
        beams = [(0.0, [self.sos_token_id])]
        completed_beams = []
        
        for step in range(self.max_len):
            candidates = []
            
            for score, sequence in beams:
                if sequence[-1] == self.eos_token_id:
                    completed_beams.append((score, sequence))
                    continue
                    
                # Create target tensor
                tgt = torch.tensor([sequence], device=device)
                tgt_mask = create_combined_mask(tgt, self.pad_token_id)
                
                # Decode
                with torch.no_grad():
                    decoder_output, _, _ = self.model.decode(
                        tgt, encoder_output, tgt_mask, src_mask
                    )
                    
                # Get logits for next token
                logits = self.model.output_projection(decoder_output[0, -1])
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get top k candidates
                top_log_probs, top_indices = log_probs.topk(self.beam_size)
                
                for i in range(self.beam_size):
                    token_id = top_indices[i].item()
                    token_score = top_log_probs[i].item()
                    new_score = score + token_score
                    new_sequence = sequence + [token_id]
                    
                    heappush(candidates, (-new_score, new_sequence))
            
            # Select top beams
            beams = []
            for _ in range(min(self.beam_size, len(candidates))):
                if candidates:
                    neg_score, sequence = heappop(candidates)
                    beams.append((-neg_score, sequence))
            
            if not beams:
                break
        
        # Add remaining beams to completed
        for score, sequence in beams:
            completed_beams.append((score, sequence))
        
        if not completed_beams:
            return torch.tensor([self.sos_token_id, self.eos_token_id]), 0.0
            
        # Return best sequence
        best_score, best_sequence = max(completed_beams, key=lambda x: x[0])
        return torch.tensor(best_sequence), best_score


def greedy_decode(
    model: torch.nn.Module,
    src: torch.Tensor,
    max_len: int = 64,
    sos_token_id: int = 1,
    eos_token_id: int = 2,
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    Greedy decoding for sequence generation.
    
    Args:
        model: Transformer model
        src: Source sequence of shape (batch_size, src_len)
        max_len: Maximum generation length
        sos_token_id: Start-of-sequence token ID
        eos_token_id: End-of-sequence token ID
        pad_token_id: Padding token ID
        
    Returns:
        Generated sequences of shape (batch_size, tgt_len)
    """
    device = src.device
    batch_size = src.size(0)
    
    # Create source mask
    src_mask = create_padding_mask(src, pad_token_id)
    
    # Encode source
    with torch.no_grad():
        encoder_output, _ = model.encode(src, src_mask)
    
    # Initialize target sequences with SOS token
    generated = torch.full((batch_size, 1), sos_token_id, device=device, dtype=torch.long)
    
    for _ in range(max_len - 1):
        # Create target mask
        tgt_mask = create_combined_mask(generated, pad_token_id)
        
        # Decode
        with torch.no_grad():
            decoder_output, _, _ = model.decode(
                generated, encoder_output, tgt_mask, src_mask
            )
            
        # Get logits for next token
        logits = model.output_projection(decoder_output[:, -1])  # (batch_size, vocab_size)
        
        # Greedy selection
        next_tokens = logits.argmax(dim=-1, keepdim=True)  # (batch_size, 1)
        
        # Append to generated sequence
        generated = torch.cat([generated, next_tokens], dim=1)
        
        # Stop if all sequences generated EOS
        if (next_tokens.squeeze(1) == eos_token_id).all():
            break
    
    return generated


def evaluate_generation(
    model: torch.nn.Module,
    dataset: ToyDataset,
    task_type: str,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    use_beam_search: bool = False,
    beam_size: int = 5
) -> dict:
    """
    Evaluate generation quality on test samples.
    
    Args:
        model: Transformer model
        dataset: Toy dataset generator
        task_type: 'copy' or 'reverse'
        num_samples: Number of samples to evaluate
        batch_size: Batch size for evaluation
        device: Device to run on
        use_beam_search: Whether to use beam search
        beam_size: Beam size for beam search
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    correct_sequences = 0
    total_sequences = 0
    
    if use_beam_search:
        decoder = BeamSearchDecoder(model, beam_size=beam_size)
    
    for _ in range(num_samples // batch_size):
        # Generate test batch
        if task_type == 'copy':
            src, expected_tgt = dataset.generate_copy_task(batch_size)
        else:  # reverse
            src, expected_tgt = dataset.generate_reverse_task(batch_size)
            
        src = src.to(device)
        expected_tgt = expected_tgt.to(device)
        
        if use_beam_search and batch_size == 1:
            # Beam search (only supports batch_size=1)
            src_mask = create_padding_mask(src)
            generated_tgt, _ = decoder.decode(src, src_mask)
            generated_tgt = generated_tgt.unsqueeze(0)
        else:
            # Greedy decoding
            generated_tgt = greedy_decode(model, src)
        
        # Compare with expected output
        for i in range(batch_size):
            # Find actual lengths (exclude padding)
            expected_len = (expected_tgt[i] != 0).sum().item()
            generated_len = (generated_tgt[i] != 0).sum().item()
            
            # Compare sequences (excluding SOS/EOS tokens for fair comparison)
            expected_seq = expected_tgt[i][1:expected_len-1]  # Remove SOS and EOS
            generated_seq = generated_tgt[i][1:generated_len-1] if generated_len > 2 else torch.tensor([], device=device)
            
            # Ensure both tensors are on the same device
            expected_seq = expected_seq.to(device)
            generated_seq = generated_seq.to(device)
            
            if torch.equal(expected_seq, generated_seq):
                correct_sequences += 1
            total_sequences += 1
    
    accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0.0
    
    return {
        'sequence_accuracy': accuracy,
        'correct_sequences': correct_sequences,
        'total_sequences': total_sequences
    }


def interactive_demo(
    model: torch.nn.Module,
    dataset: ToyDataset,
    device: torch.device,
    use_beam_search: bool = False,
    beam_size: int = 5
):
    """
    Interactive demo for manual testing.
    """
    model.eval()
    
    if use_beam_search:
        decoder = BeamSearchDecoder(model, beam_size=beam_size)
        print(f"Using beam search with beam size {beam_size}")
    else:
        print("Using greedy decoding")
    
    print("\\nInteractive demo - Enter 'quit' to exit")
    print("Enter sequences as space-separated integers (3-99)")
    print("Example: 5 10 15 20")
    
    while True:
        try:
            user_input = input("\\nInput sequence: ").strip()
            if user_input.lower() == 'quit':
                break
                
            # Parse input sequence
            tokens = list(map(int, user_input.split()))
            
            # Create source tensor
            src_seq = tokens + [2]  # Add EOS token
            src = torch.tensor([src_seq], device=device)
            
            print(f"Source: {tokens}")
            
            if use_beam_search:
                src_mask = create_padding_mask(src)
                generated, score = decoder.decode(src, src_mask)
                generated_tokens = generated[1:-1].tolist()  # Remove SOS/EOS
                print(f"Generated (beam search, score={score:.3f}): {generated_tokens}")
            else:
                generated = greedy_decode(model, src)
                generated_tokens = generated[0][1:].tolist()
                # Find EOS and truncate
                if 2 in generated_tokens:
                    eos_idx = generated_tokens.index(2)
                    generated_tokens = generated_tokens[:eos_idx]
                print(f"Generated (greedy): {generated_tokens}")
                
        except (ValueError, KeyboardInterrupt):
            print("Invalid input. Please enter space-separated integers.")
        except KeyboardInterrupt:
            break
    
    print("Demo ended.")


def main():
    parser = argparse.ArgumentParser(description='Decode sequences with trained Transformer')
    
    # Model parameters (should match training config)
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--use_geglu', action='store_true', help='Use GEGLU in FFN')
    
    # Dataset parameters
    parser.add_argument('--vocab_size', type=int, default=100, help='Vocabulary size')
    parser.add_argument('--max_seq_len', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--seq_len_range', type=str, default='5,20', help='Sequence length range (min,max)')
    parser.add_argument('--task', type=str, default='copy', choices=['copy', 'reverse'], help='Task type')
    
    # Decoding parameters
    parser.add_argument('--use_beam_search', action='store_true', help='Use beam search decoding')
    parser.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search')
    parser.add_argument('--max_decode_len', type=int, default=64, help='Maximum decoding length')
    
    # Evaluation parameters
    parser.add_argument('--eval_samples', type=int, default=1000, help='Number of samples for evaluation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    
    # Other parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--interactive', action='store_true', help='Run interactive demo')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Parse sequence length range
    seq_len_min, seq_len_max = map(int, args.seq_len_range.split(','))
    
    # Create dataset
    dataset = ToyDataset(
        vocab_size=args.vocab_size,
        seq_len_range=(seq_len_min, seq_len_max)
    )
    
    # Create model
    model = Transformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        use_geglu=args.use_geglu,
        pad_token_id=0
    ).to(device)
    
    # Load checkpoint
    try:
        load_checkpoint(model, None, args.checkpoint, device)
        print(f"Loaded model from {args.checkpoint}")
    except FileNotFoundError:
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    
    if args.interactive:
        # Run interactive demo
        interactive_demo(model, dataset, device, args.use_beam_search, args.beam_size)
    else:
        # Run evaluation
        print(f"Evaluating on {args.eval_samples} samples...")
        
        if args.use_beam_search and args.batch_size > 1:
            print("Warning: Beam search only supports batch_size=1. Setting batch_size=1.")
            args.batch_size = 1
        
        results = evaluate_generation(
            model, dataset, args.task, args.eval_samples, args.batch_size,
            device, args.use_beam_search, args.beam_size
        )
        
        print(f"\\nEvaluation Results ({args.task} task):")
        print(f"Sequence Accuracy: {results['sequence_accuracy']:.4f}")
        print(f"Correct Sequences: {results['correct_sequences']}/{results['total_sequences']}")
        
        decode_method = "beam search" if args.use_beam_search else "greedy"
        print(f"Decoding method: {decode_method}")


if __name__ == '__main__':
    main()
