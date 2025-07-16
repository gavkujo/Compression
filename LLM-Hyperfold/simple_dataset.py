#!/usr/bin/env python3

"""Test script to create datasets directly"""

import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

def create_math_dataset(num_samples=1500):
    """Create exactly 1500 math samples"""
    print(f"Creating {num_samples} math samples...")
    
    prompts = []
    outputs = []
    expert_types = []
    
    # Simple arithmetic - exactly 1500 samples
    for i in range(num_samples):
        a, b = np.random.randint(1, 1000, 2)
        op = np.random.choice(['+', '-', '*'])
        
        if op == '+':
            result = a + b
            prompts.append(f"What is {a} + {b}?")
            outputs.append(f"The answer is {result}.")
        elif op == '-':
            result = a - b
            prompts.append(f"What is {a} - {b}?")
            outputs.append(f"The answer is {result}.")
        else:  # multiplication
            result = a * b
            prompts.append(f"What is {a} × {b}?")
            outputs.append(f"The answer is {result}.")
        
        expert_types.append("math")
    
    return prompts, outputs, expert_types

def create_code_dataset(num_samples=1500):
    """Create exactly 1500 code samples"""
    print(f"Creating {num_samples} code samples...")
    
    prompts = []
    outputs = []
    expert_types = []
    
    templates = [
        ("Write a function to add two numbers", "def add(a, b):\n    return a + b"),
        ("Write a function to multiply two numbers", "def multiply(a, b):\n    return a * b"),
        ("Write a function to check if number is even", "def is_even(n):\n    return n % 2 == 0"),
        ("Create an empty list", "my_list = []"),
        ("Add element to list", "my_list.append(element)"),
    ]
    
    for i in range(num_samples):
        template_idx = i % len(templates)
        prompt, output = templates[template_idx]
        prompts.append(prompt)
        outputs.append(output)
        expert_types.append("code")
    
    return prompts, outputs, expert_types

def create_creative_dataset(num_samples=1500):
    """Create exactly 1500 creative samples"""
    print(f"Creating {num_samples} creative samples...")
    
    prompts = []
    outputs = []
    expert_types = []
    
    templates = [
        ("Write a short story about a cat", "Once upon a time, there was a clever cat named Whiskers who loved to explore."),
        ("Write a haiku about nature", "Cherry blossoms fall\nGentle breeze carries petals\nSpring's fleeting beauty"),
        ("Describe a magical forest", "Ancient trees with silver bark stretch toward starlit skies, while glowing mushrooms light the winding paths."),
    ]
    
    for i in range(num_samples):
        template_idx = i % len(templates)
        prompt, output = templates[template_idx]
        prompts.append(prompt)
        outputs.append(output)
        expert_types.append("creative")
    
    return prompts, outputs, expert_types

def create_general_dataset(num_samples=1500):
    """Create exactly 1500 general samples"""
    print(f"Creating {num_samples} general samples...")
    
    prompts = []
    outputs = []
    expert_types = []
    
    templates = [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the largest ocean?", "The Pacific Ocean is the largest ocean."),
        ("When did World War II end?", "World War II ended in 1945."),
        ("What does CPU stand for?", "CPU stands for Central Processing Unit."),
        ("Who invented the telephone?", "Alexander Graham Bell invented the telephone."),
    ]
    
    for i in range(num_samples):
        template_idx = i % len(templates)
        prompt, output = templates[template_idx]
        prompts.append(prompt)
        outputs.append(output)
        expert_types.append("general")
    
    return prompts, outputs, expert_types

def main():
    """Create all datasets"""
    print("📊 SIMPLE DATASET CREATOR")
    print("=" * 40)
    
    # Create output directory
    os.makedirs("datasets", exist_ok=True)
    
    # Create each dataset
    datasets = {
        'math': create_math_dataset(),
        'code': create_code_dataset(), 
        'creative': create_creative_dataset(),
        'general': create_general_dataset()
    }
    
    # Save each dataset
    for expert_type, (prompts, outputs, expert_types) in datasets.items():
        df = pd.DataFrame({
            'prompt': prompts,
            'output': outputs,
            'expert_type': expert_types
        })
        
        path = f"datasets/{expert_type}_dataset.csv"
        df.to_csv(path, index=False)
        print(f"✅ {expert_type.title()} dataset saved: {path} ({len(df)} samples)")
    
    # Create combined dataset
    all_prompts = []
    all_outputs = []
    all_expert_types = []
    
    for expert_type, (prompts, outputs, expert_types) in datasets.items():
        all_prompts.extend(prompts)
        all_outputs.extend(outputs)
        all_expert_types.extend(expert_types)
    
    combined_df = pd.DataFrame({
        'prompt': all_prompts,
        'output': all_outputs,
        'expert_type': all_expert_types
    })
    
    combined_path = "datasets/combined_dataset.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"✅ Combined dataset saved: {combined_path} ({len(combined_df)} samples)")
    print(f"📊 Total: {len(combined_df)} samples (1500 per expert)")

if __name__ == "__main__":
    main()
