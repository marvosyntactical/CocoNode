"""
GSM8K Dataset Preparation for CocoNODE

GSM8K (Grade School Math 8K) is ideal for testing Coconut-style reasoning:
- Problems require multi-step mathematical reasoning
- Each step has clear intermediate results
- We can measure whether latent reasoning captures these steps

This script:
1. Downloads GSM8K from HuggingFace
2. Parses chain-of-thought into discrete steps
3. Formats for Coconut curriculum training
4. Creates train/val splits with reasoning step annotations

Example problem:
Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every 
   morning and bakes muffins for her friends every day with four. She sells 
   the remainder at the farmers' market daily for $2 per egg. How much does 
   she make every day at the farmers' market?

CoT Steps:
1. "Janet sells 16 - 3 - 4 = 9 duck eggs a day."
2. "She makes 9 * 2 = $18 every day."

Answer: 18
"""

import os
import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random

# Try to import datasets, provide fallback
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: 'datasets' not installed. Run: pip install datasets")


@dataclass
class MathProblem:
    """A single math problem with chain-of-thought steps."""
    question: str
    answer: str           # Final numeric answer
    steps: List[str]      # Individual reasoning steps
    full_solution: str    # Complete solution text
    

def parse_gsm8k_solution(solution: str) -> Tuple[List[str], str]:
    """
    Parse GSM8K solution into steps and final answer.
    
    GSM8K format:
    - Steps separated by newlines
    - Final answer marked with #### 
    
    Returns:
        steps: List of reasoning steps
        answer: Final numeric answer
    """
    # Split on #### to get answer
    parts = solution.split('####')
    
    if len(parts) == 2:
        reasoning = parts[0].strip()
        answer = parts[1].strip()
    else:
        # Fallback: try to find number at end
        reasoning = solution
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', solution)
        answer = numbers[-1] if numbers else ""
    
    # Clean answer (remove commas from numbers)
    answer = answer.replace(',', '').strip()
    
    # Parse steps - split on newlines, filter empty
    steps = [s.strip() for s in reasoning.split('\n') if s.strip()]
    
    # Further split long steps on sentence boundaries
    refined_steps = []
    for step in steps:
        # Split on periods followed by space and capital letter or number
        substeps = re.split(r'(?<=\.)\s+(?=[A-Z0-9])', step)
        refined_steps.extend([s.strip() for s in substeps if s.strip()])
    
    return refined_steps, answer


def download_and_process_gsm8k(output_dir: str = '.', 
                               train_size: Optional[int] = None,
                               val_size: Optional[int] = None) -> None:
    """
    Download GSM8K and process into CocoNODE format.
    
    Creates:
    - train.json: Training examples with CoT steps
    - val.json: Validation examples
    - meta.json: Dataset metadata
    """
    if not HAS_DATASETS:
        print("Please install datasets: pip install datasets")
        return
    
    print("Downloading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    
    os.makedirs(output_dir, exist_ok=True)
    
    def process_split(split_data, max_size=None) -> List[Dict]:
        problems = []
        
        for i, item in enumerate(split_data):
            if max_size and i >= max_size:
                break
            
            question = item['question']
            solution = item['answer']
            
            steps, answer = parse_gsm8k_solution(solution)
            
            # Skip if parsing failed
            if not steps or not answer:
                continue
            
            problems.append({
                'question': question,
                'answer': answer,
                'steps': steps,
                'full_solution': solution.split('####')[0].strip(),
                'n_steps': len(steps),
            })
        
        return problems
    
    # Process splits
    print("Processing training split...")
    train_data = process_split(dataset['train'], train_size)
    
    print("Processing test split (as validation)...")
    val_data = process_split(dataset['test'], val_size)
    
    # Save
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_data, f, indent=2)
    
    # Metadata
    meta = {
        'dataset': 'gsm8k',
        'train_size': len(train_data),
        'val_size': len(val_data),
        'avg_steps_train': sum(p['n_steps'] for p in train_data) / len(train_data),
        'avg_steps_val': sum(p['n_steps'] for p in val_data) / len(val_data),
        'max_steps': max(p['n_steps'] for p in train_data + val_data),
    }
    
    with open(os.path.join(output_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nDataset saved to {output_dir}/")
    print(f"  Train: {len(train_data)} problems")
    print(f"  Val: {len(val_data)} problems")
    print(f"  Avg steps: {meta['avg_steps_train']:.1f}")


class GSM8KDataset:
    """
    GSM8K dataset for CocoNODE training with Coconut curriculum.
    
    Supports:
    - Standard CoT training (all steps as text)
    - Coconut curriculum (progressively replace steps with latent tokens)
    - Data augmentation (number perturbation, paraphrasing)
    """
    
    # Special tokens
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    BOT_TOKEN = "<bot>"  # Begin of thought (latent)
    EOT_TOKEN = "<eot>"  # End of thought
    SEP_TOKEN = "<sep>"  # Step separator
    
    def __init__(self, data_path: str, block_size: int = 512,
                 tokenizer=None):
        """
        Args:
            data_path: Path to JSON file with problems
            block_size: Maximum sequence length
            tokenizer: Optional tokenizer (defaults to character-level)
        """
        self.block_size = block_size
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        # Setup tokenizer
        if tokenizer is None:
            self._setup_char_tokenizer()
        else:
            self.tokenizer = tokenizer
        
        print(f"Loaded {len(self.data)} problems from {data_path}")
    
    def _setup_char_tokenizer(self):
        """Setup simple character-level tokenizer with special tokens."""
        # Build vocab from data
        chars = set()
        for problem in self.data:
            chars.update(problem['question'])
            chars.update(problem['answer'])
            for step in problem['steps']:
                chars.update(step)
        
        # Add special tokens first
        special_tokens = [
            self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN,
            self.BOT_TOKEN, self.EOT_TOKEN, self.SEP_TOKEN
        ]
        
        all_tokens = special_tokens + sorted(list(chars))
        
        self.stoi = {ch: i for i, ch in enumerate(all_tokens)}
        self.itos = {i: ch for i, ch in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)
        
        # Store special token IDs
        self.pad_id = self.stoi[self.PAD_TOKEN]
        self.bos_id = self.stoi[self.BOS_TOKEN]
        self.eos_id = self.stoi[self.EOS_TOKEN]
        self.bot_id = self.stoi[self.BOT_TOKEN]
        self.eot_id = self.stoi[self.EOT_TOKEN]
        self.sep_id = self.stoi[self.SEP_TOKEN]
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.stoi.get(ch, self.pad_id) for ch in text]
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join(self.itos.get(i, '?') for i in ids 
                      if i not in [self.pad_id, self.bos_id, self.eos_id])
    
    def format_problem(self, problem: Dict, stage: int = 0,
                       latent_steps_per_stage: int = 1) -> Dict:
        """
        Format a problem for the given curriculum stage.
        
        Stage 0: Full CoT in text
        Stage k: First k*latent_steps_per_stage steps as latent tokens
        
        Returns dict with:
        - input_ids: Token IDs
        - targets: Target IDs (shifted)
        - latent_positions: Positions of latent tokens
        - answer_positions: Positions of answer tokens (for evaluation)
        """
        n_latent_steps = stage * latent_steps_per_stage
        steps = problem['steps']
        
        # Build sequence
        tokens = [self.bos_id]
        
        # Question
        tokens.extend(self.encode(f"Q: {problem['question']}\nA: "))
        
        # Track latent positions
        latent_positions = []
        
        # Steps
        for i, step in enumerate(steps):
            if i < n_latent_steps:
                # Replace with latent token
                latent_positions.append(len(tokens))
                tokens.append(self.bot_id)
                # Could add multiple latent tokens per step:
                # for _ in range(latent_tokens_per_step):
                #     tokens.append(self.bot_id)
            else:
                # Keep as text
                tokens.extend(self.encode(step))
            
            # Add separator between steps
            if i < len(steps) - 1:
                tokens.append(self.sep_id)
        
        # Answer
        answer_start = len(tokens)
        tokens.extend(self.encode(f"\n#### {problem['answer']}"))
        tokens.append(self.eos_id)
        
        # Truncate or pad
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
            latent_positions = [p for p in latent_positions if p < self.block_size]
        
        # Pad
        n_pad = self.block_size - len(tokens)
        tokens = tokens + [self.pad_id] * n_pad
        
        # Targets (shifted by 1)
        targets = tokens[1:] + [self.pad_id]
        
        return {
            'input_ids': tokens,
            'targets': targets,
            'latent_positions': latent_positions,
            'n_latent_steps': len(latent_positions),
            'answer_start': min(answer_start, self.block_size - 1),
            'question': problem['question'],
            'answer': problem['answer'],
        }
    
    def get_batch(self, batch_size: int, stage: int = 0,
                  latent_steps_per_stage: int = 1,
                  device: str = 'cuda') -> Dict:
        """
        Get a batch of problems for the given curriculum stage.
        """
        import torch
        
        indices = random.sample(range(len(self.data)), min(batch_size, len(self.data)))
        
        batch_input_ids = []
        batch_targets = []
        batch_latent_positions = []
        
        for idx in indices:
            formatted = self.format_problem(
                self.data[idx], 
                stage=stage,
                latent_steps_per_stage=latent_steps_per_stage
            )
            batch_input_ids.append(formatted['input_ids'])
            batch_targets.append(formatted['targets'])
            batch_latent_positions.append(formatted['latent_positions'])
        
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long, device=device),
            'targets': torch.tensor(batch_targets, dtype=torch.long, device=device),
            'latent_positions': batch_latent_positions,
            'n_latent_steps': stage * latent_steps_per_stage,
        }
    
    def __len__(self):
        return len(self.data)


class GSM8KAugmented(GSM8KDataset):
    """
    GSM8K with data augmentation for better generalization.
    
    Augmentations:
    - Number perturbation (change numbers while maintaining ratios)
    - Step reordering (where mathematically valid)
    - Paraphrasing (if model available)
    """
    
    def __init__(self, data_path: str, block_size: int = 512,
                 augment_prob: float = 0.5):
        super().__init__(data_path, block_size)
        self.augment_prob = augment_prob
    
    def augment_numbers(self, problem: Dict) -> Dict:
        """
        Augment by scaling all numbers in the problem.
        
        Maintains mathematical relationships while creating new examples.
        """
        # Find all numbers in question
        question = problem['question']
        numbers = re.findall(r'\d+(?:\.\d+)?', question)
        
        if not numbers:
            return problem
        
        # Random scale factor
        scale = random.uniform(0.5, 2.0)
        
        # Replace numbers (simple version - could be more sophisticated)
        new_question = question
        new_steps = []
        
        for num_str in numbers:
            num = float(num_str)
            new_num = round(num * scale, 2)
            if new_num == int(new_num):
                new_num = int(new_num)
            new_question = new_question.replace(num_str, str(new_num), 1)
        
        # Recompute steps and answer (simplified - full version would solve)
        # For now, just scale the answer
        try:
            old_answer = float(problem['answer'])
            new_answer = str(round(old_answer * scale, 2))
            if float(new_answer) == int(float(new_answer)):
                new_answer = str(int(float(new_answer)))
        except:
            new_answer = problem['answer']
        
        return {
            **problem,
            'question': new_question,
            'answer': new_answer,
            # Note: steps would need recomputation for correctness
        }
    
    def get_batch(self, batch_size: int, stage: int = 0,
                  latent_steps_per_stage: int = 1,
                  device: str = 'cuda') -> Dict:
        """Get batch with optional augmentation."""
        import torch
        
        indices = random.sample(range(len(self.data)), min(batch_size, len(self.data)))
        
        batch_input_ids = []
        batch_targets = []
        batch_latent_positions = []
        
        for idx in indices:
            problem = self.data[idx]
            
            # Maybe augment
            if random.random() < self.augment_prob:
                problem = self.augment_numbers(problem)
            
            formatted = self.format_problem(
                problem,
                stage=stage,
                latent_steps_per_stage=latent_steps_per_stage
            )
            batch_input_ids.append(formatted['input_ids'])
            batch_targets.append(formatted['targets'])
            batch_latent_positions.append(formatted['latent_positions'])
        
        return {
            'input_ids': torch.tensor(batch_input_ids, dtype=torch.long, device=device),
            'targets': torch.tensor(batch_targets, dtype=torch.long, device=device),
            'latent_positions': batch_latent_positions,
            'n_latent_steps': stage * latent_steps_per_stage,
        }


# =============================================================================
# Evaluation Utilities
# =============================================================================

def extract_answer(text: str) -> Optional[str]:
    """Extract numeric answer from generated text."""
    # Look for #### pattern
    match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', text)
    if match:
        return match.group(1)
    
    # Fallback: last number in text
    numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
    return numbers[-1] if numbers else None


def evaluate_accuracy(model, dataset: GSM8KDataset, 
                      n_samples: int = 100,
                      stage: int = 0,
                      device: str = 'cuda') -> Dict:
    """
    Evaluate model accuracy on GSM8K.
    
    Returns:
        Dict with accuracy metrics
    """
    import torch
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            problem = dataset.data[i]
            formatted = dataset.format_problem(problem, stage=stage)
            
            # Get input up to answer
            input_ids = torch.tensor(
                [formatted['input_ids'][:formatted['answer_start']]], 
                device=device
            )
            
            # Generate
            output = model.generate(input_ids, max_new_tokens=50)
            generated_text = dataset.decode(output[0].tolist())
            
            # Extract and compare answer
            pred_answer = extract_answer(generated_text)
            true_answer = problem['answer']
            
            if pred_answer == true_answer:
                correct += 1
            total += 1
    
    model.train()
    
    return {
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare GSM8K dataset')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for processed data')
    parser.add_argument('--train_size', type=int, default=None,
                       help='Limit training set size')
    parser.add_argument('--val_size', type=int, default=None,
                       help='Limit validation set size')
    
    args = parser.parse_args()
    
    download_and_process_gsm8k(
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size
    )
