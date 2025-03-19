#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prediction script for HMM-based Named Entity Recognition on Twitter data.
"""

import os
import sys
import argparse
import json
import pickle
import time

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hmm_tagger import HMMTagger
from src.data_utils import preprocess_data


def predict_file(model, input_file):
    """
    Predict NER tags for sentences in a file.
    
    Args:
        model: The trained HMMTagger model
        input_file: Path to the input file with sentences (one token per line, blank lines between sentences)
        
    Returns:
        List of sentences with predicted tags
    """
    data = preprocess_data(input_file)
    sentences = [[word for word, _ in sentence] for sentence in data]
    
    predicted_tags = []
    for sentence in sentences:
        tags = model.viterbi_decode(sentence)
        predicted_tags.append(tags)
    
    return sentences, predicted_tags


def main():
    """Main function to make predictions using a trained HMM model."""
    parser = argparse.ArgumentParser(description='Make predictions with trained HMM-based NER model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input file to make predictions on')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the output file to save predictions')
    parser.add_argument('--format', type=str, default='columns',
                        choices=['columns', 'conll', 'json'],
                        help='Format for output file')
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = HMMTagger()
    model.load_model(args.model)
    
    # Make predictions
    print(f"Making predictions on {args.input}...")
    start_time = time.time()
    sentences, predictions = predict_file(model, args.input)
    prediction_time = time.time() - start_time
    print(f"Predictions completed in {prediction_time:.2f} seconds")
    
    # Save predictions
    print(f"Saving predictions to {args.output}...")
    if args.format == 'columns':
        with open(args.output, 'w', encoding='utf-8') as f:
            for sentence, tags in zip(sentences, predictions):
                for word, tag in zip(sentence, tags):
                    f.write(f"{word}\t{tag}\n")
                f.write("\n")
    
    elif args.format == 'conll':
        with open(args.output, 'w', encoding='utf-8') as f:
            for i, (sentence, tags) in enumerate(zip(sentences, predictions)):
                for j, (word, tag) in enumerate(zip(sentence, tags)):
                    f.write(f"{j+1}\t{word}\t_\t_\t_\t_\t_\t_\t_\t{tag}\n")
                f.write("\n")
    
    elif args.format == 'json':
        results = []
        for sentence, tags in zip(sentences, predictions):
            results.append({
                'words': sentence,
                'tags': tags
            })
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()
