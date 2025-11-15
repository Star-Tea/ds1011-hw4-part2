"""
Script to compute data statistics for Q4
Run this to get the statistics needed for your written report
"""

import os
from transformers import T5TokenizerFast
from collections import Counter
import numpy as np

def load_lines(path):
    """Load lines from a file."""
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    return lines

def compute_statistics_before_preprocessing(data_folder='data'):
    """
    Compute statistics BEFORE any preprocessing.
    This uses the raw .nl and .sql files.
    """
    print("="*80)
    print("STATISTICS BEFORE PREPROCESSING")
    print("="*80)
    
    # Initialize tokenizer for token counting
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    stats = {}
    
    for split in ['train', 'dev']:
        print(f"\n--- {split.upper()} SET ---")
        
        # Load natural language queries
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_queries = load_lines(nl_path)
        
        # Load SQL queries
        sql_path = os.path.join(data_folder, f'{split}.sql')
        sql_queries = load_lines(sql_path)
        
        # Number of examples
        n_examples = len(nl_queries)
        print(f"Number of examples: {n_examples}")
        
        # Tokenize to count tokens
        nl_tokens_list = [tokenizer.tokenize(q) for q in nl_queries]
        sql_tokens_list = [tokenizer.tokenize(q) for q in sql_queries]
        
        # Mean sentence length (in tokens)
        nl_lengths = [len(tokens) for tokens in nl_tokens_list]
        mean_nl_length = np.mean(nl_lengths)
        print(f"Mean NL query length (tokens): {mean_nl_length:.2f}")
        
        # Mean SQL query length (in tokens)
        sql_lengths = [len(tokens) for tokens in sql_tokens_list]
        mean_sql_length = np.mean(sql_lengths)
        print(f"Mean SQL query length (tokens): {mean_sql_length:.2f}")
        
        # Vocabulary size for natural language
        nl_vocab = set()
        for tokens in nl_tokens_list:
            nl_vocab.update(tokens)
        nl_vocab_size = len(nl_vocab)
        print(f"Vocabulary size (NL): {nl_vocab_size}")
        
        # Vocabulary size for SQL
        sql_vocab = set()
        for tokens in sql_tokens_list:
            sql_vocab.update(tokens)
        sql_vocab_size = len(sql_vocab)
        print(f"Vocabulary size (SQL): {sql_vocab_size}")
        
        # Store statistics
        stats[split] = {
            'n_examples': n_examples,
            'mean_nl_length': mean_nl_length,
            'mean_sql_length': mean_sql_length,
            'nl_vocab_size': nl_vocab_size,
            'sql_vocab_size': sql_vocab_size
        }
    
    # Test set (only NL queries, no SQL)
    print(f"\n--- TEST SET ---")
    test_nl_path = os.path.join(data_folder, 'test.nl')
    test_nl_queries = load_lines(test_nl_path)
    n_test = len(test_nl_queries)
    print(f"Number of examples: {n_test}")
    
    test_nl_tokens = [tokenizer.tokenize(q) for q in test_nl_queries]
    test_nl_lengths = [len(tokens) for tokens in test_nl_tokens]
    mean_test_nl_length = np.mean(test_nl_lengths)
    print(f"Mean NL query length (tokens): {mean_test_nl_length:.2f}")
    
    stats['test'] = {
        'n_examples': n_test,
        'mean_nl_length': mean_test_nl_length
    }
    
    return stats

def compute_statistics_after_preprocessing(data_folder='data'):
    """
    Compute statistics AFTER preprocessing.
    This shows what the model actually sees after tokenization with special tokens.
    """
    print("\n" + "="*80)
    print("STATISTICS AFTER PREPROCESSING")
    print("="*80)
    print(f"Model: google-t5/t5-small")
    print(f"Tokenizer: T5TokenizerFast")
    
    # Initialize tokenizer
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    
    stats = {}
    
    for split in ['train', 'dev']:
        print(f"\n--- {split.upper()} SET ---")
        
        # Load data
        nl_path = os.path.join(data_folder, f'{split}.nl')
        nl_queries = load_lines(nl_path)
        
        sql_path = os.path.join(data_folder, f'{split}.sql')
        sql_queries = load_lines(sql_path)
        
        n_examples = len(nl_queries)
        print(f"Number of examples: {n_examples}")
        
        # Tokenize WITH special tokens (as the model sees it)
        # Encoder input
        nl_tokenized = [tokenizer(q, add_special_tokens=True) for q in nl_queries]
        nl_lengths = [len(tok['input_ids']) for tok in nl_tokenized]
        mean_nl_length = np.mean(nl_lengths)
        print(f"Mean encoder input length (with special tokens): {mean_nl_length:.2f}")
        
        # Decoder input (with BOS token)
        sql_with_bos = [f"<extra_id_0> {q}" for q in sql_queries]
        sql_tokenized = [tokenizer(q, add_special_tokens=True) for q in sql_with_bos]
        sql_lengths = [len(tok['input_ids']) for tok in sql_tokenized]
        mean_sql_length = np.mean(sql_lengths)
        print(f"Mean decoder input length (with BOS + special tokens): {mean_sql_length:.2f}")
        
        # Vocabulary is the T5 tokenizer vocabulary
        tokenizer_vocab_size = len(tokenizer)
        print(f"Tokenizer vocabulary size: {tokenizer_vocab_size}")
        
        # Actual tokens used in this dataset
        all_nl_tokens = set()
        for tok in nl_tokenized:
            all_nl_tokens.update(tok['input_ids'])
        print(f"Unique tokens in NL queries: {len(all_nl_tokens)}")
        
        all_sql_tokens = set()
        for tok in sql_tokenized:
            all_sql_tokens.update(tok['input_ids'])
        print(f"Unique tokens in SQL queries: {len(all_sql_tokens)}")
        
        stats[split] = {
            'n_examples': n_examples,
            'mean_encoder_length': mean_nl_length,
            'mean_decoder_length': mean_sql_length,
            'tokenizer_vocab_size': tokenizer_vocab_size,
            'unique_nl_tokens': len(all_nl_tokens),
            'unique_sql_tokens': len(all_sql_tokens)
        }
    
    # Test set
    print(f"\n--- TEST SET ---")
    test_nl_path = os.path.join(data_folder, 'test.nl')
    test_nl_queries = load_lines(test_nl_path)
    n_test = len(test_nl_queries)
    print(f"Number of examples: {n_test}")
    
    test_nl_tokenized = [tokenizer(q, add_special_tokens=True) for q in test_nl_queries]
    test_nl_lengths = [len(tok['input_ids']) for tok in test_nl_tokenized]
    mean_test_nl_length = np.mean(test_nl_lengths)
    print(f"Mean encoder input length (with special tokens): {mean_test_nl_length:.2f}")
    
    stats['test'] = {
        'n_examples': n_test,
        'mean_encoder_length': mean_test_nl_length
    }
    
    return stats

def generate_latex_tables(before_stats, after_stats):
    """
    Generate LaTeX tables for the report.
    """
    print("\n" + "="*80)
    print("LATEX TABLES FOR YOUR REPORT")
    print("="*80)
    
    print("\nTABLE 1: Data statistics BEFORE any pre-processing")
    print("-" * 80)
    print(r"""
\begin{table}[h]
\centering
\begin{tabular}{lcc}
\hline
\textbf{Statistics Name} & \textbf{Train} & \textbf{Dev} \\
\hline""")
    print(f"Number of examples & {before_stats['train']['n_examples']} & {before_stats['dev']['n_examples']} \\\\")
    print(f"Mean sentence length & {before_stats['train']['mean_nl_length']:.2f} & {before_stats['dev']['mean_nl_length']:.2f} \\\\")
    print(f"Mean SQL query length & {before_stats['train']['mean_sql_length']:.2f} & {before_stats['dev']['mean_sql_length']:.2f} \\\\")
    print(f"Vocabulary size (natural language) & {before_stats['train']['nl_vocab_size']} & {before_stats['dev']['nl_vocab_size']} \\\\")
    print(f"Vocabulary size (SQL) & {before_stats['train']['sql_vocab_size']} & {before_stats['dev']['sql_vocab_size']} \\\\")
    print(r"""\hline
\end{tabular}
\caption{Data statistics before any pre-processing using T5 tokenizer.}
\end{table}
""")
    
    print("\nTABLE 2: Data statistics AFTER pre-processing")
    print("-" * 80)
    print(r"""
\begin{table}[h]
\centering
\begin{tabular}{lcc}
\hline
\textbf{Statistics Name} & \textbf{Train} & \textbf{Dev} \\
\hline
\multicolumn{3}{l}{\textbf{Model name:} google-t5/t5-small} \\
\hline""")
    print(f"Mean encoder input length & {after_stats['train']['mean_encoder_length']:.2f} & {after_stats['dev']['mean_encoder_length']:.2f} \\\\")
    print(f"Mean decoder input length & {after_stats['train']['mean_decoder_length']:.2f} & {after_stats['dev']['mean_decoder_length']:.2f} \\\\")
    print(f"Tokenizer vocabulary size & {after_stats['train']['tokenizer_vocab_size']} & {after_stats['dev']['tokenizer_vocab_size']} \\\\")
    print(f"Unique tokens (NL) & {after_stats['train']['unique_nl_tokens']} & {after_stats['dev']['unique_nl_tokens']} \\\\")
    print(f"Unique tokens (SQL) & {after_stats['train']['unique_sql_tokens']} & {after_stats['dev']['unique_sql_tokens']} \\\\")
    print(r"""\hline
\end{tabular}
\caption{Data statistics after pre-processing. Lengths include special tokens and BOS token for decoder.}
\end{table}
""")

def main():
    """
    Main function to compute and display all statistics.
    """
    print("\nComputing data statistics for Q4...")
    print("Make sure you have the data/ folder with train.nl, train.sql, dev.nl, dev.sql, test.nl")
    print()
    
    # Check if data folder exists
    if not os.path.exists('data'):
        print("ERROR: 'data' folder not found!")
        print("Please make sure you have the data folder in the current directory.")
        return
    
    # Compute statistics before preprocessing
    before_stats = compute_statistics_before_preprocessing()
    
    # Compute statistics after preprocessing
    after_stats = compute_statistics_after_preprocessing()
    
    # Generate LaTeX tables
    generate_latex_tables(before_stats, after_stats)
    
    print("\n" + "="*80)
    print("SUMMARY FOR QUICK REFERENCE")
    print("="*80)
    print(f"\nTrain set: {before_stats['train']['n_examples']} examples")
    print(f"Dev set: {before_stats['dev']['n_examples']} examples")
    print(f"Test set: {before_stats['test']['n_examples']} examples")
    print(f"\nMean NL length (before): Train={before_stats['train']['mean_nl_length']:.2f}, Dev={before_stats['dev']['mean_nl_length']:.2f}")
    print(f"Mean NL length (after): Train={after_stats['train']['mean_encoder_length']:.2f}, Dev={after_stats['dev']['mean_encoder_length']:.2f}")
    print(f"\nT5 tokenizer vocabulary size: {after_stats['train']['tokenizer_vocab_size']}")
    print("\nCopy the LaTeX tables above into your report!")

if __name__ == "__main__":
    main()