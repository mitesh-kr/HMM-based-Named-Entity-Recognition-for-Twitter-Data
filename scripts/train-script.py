#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for HMM-based Named Entity Recognition on Twitter data.
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
from src.data_utils import preprocess_data, analyze_dataset
from src.visualization import plot_tag_distribution


def main():
    """Main function to train the HMM tagger model."""
    parser = argparse.ArgumentParser(description='Train HMM-based NER model')
    parser.add_argument('--config', type=str, default='configs/model_configs.json',
                        help='Path to model configuration file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the data files')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save the trained models')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze the dataset before training')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load configuration
    with open(args.config, 'r') as f:
        configs = json.load(f)

    # Define file paths
    train_file = os.path.join(args.data_dir, 'train.txt')
    dev_file = os.path.join(args.data_dir, 'dev.txt')

    # Analyze datasets if requested
    if args.analyze:
        print("Analyzing training dataset...")
        train_stats = analyze_dataset(train_file, "Train")
        
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(args.output_dir, 'plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
            
        plot_tag_distribution(train_stats, "Train", plots_dir)
        
        print("Analyzing validation dataset...")
        dev_stats = analyze_dataset(dev_file, "Dev")
        plot_tag_distribution(dev_stats, "Dev", plots_dir)

    # Train models based on configurations
    for config in configs["models"]:
        model_name = config["name"]
        print(f"\n{'='*50}")
        print(f"Training model: {model_name}")
        print(f"{'='*50}")
        print(f"Configuration: Use context: {config['use_context']}, N-gram: {config['n_gram']}")

        # Process training data
        print("Processing training data...")
        train_data = preprocess_data(train_file)

        # Initialize and train model
        model = HMMTagger(use_context=config['use_context'], n_gram=config['n_gram'])
        
        start_time = time.time()
        print("Training model...")
        model.fit(train_data)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Save the trained model
        model_file = os.path.join(args.output_dir, f"model_{model_name}.pkl")
        model.save_model(model_file)
        print(f"Model saved to {model_file}")

        # Validate model
        print("\nValidating model...")
        # Process validation data
        dev_data = preprocess_data(dev_file)
        dev_words = [[word for word, _ in sentence] for sentence in dev_data]
        dev_tags = [[tag for _, tag in sentence] for sentence in dev_data]
        
        start_time = time.time()
        predicted_tags = []
        for sentence in dev_words:
            predicted_tags.append(model.viterbi_decode(sentence))
        
        # Flatten lists for evaluation
        flat_predicted = [tag for sent in predicted_tags for tag in sent]
        flat_actual = [tag for sent in dev_tags for tag in sent]
        
        validation_time = time.time() - start_time
        print(f"Validation completed in {validation_time:.2f} seconds")
        
        # Calculate accuracy
        correct = sum(1 for p, a in zip(flat_predicted, flat_actual) if p == a)
        accuracy = correct / len(flat_actual) if flat_actual else 0
        
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        # Save performance metrics
        metrics = {
            "model": model_name,
            "training_time": training_time,
            "validation_time": validation_time,
            "validation_accuracy": accuracy,
            "config": config
        }
        
        metrics_file = os.path.join(args.output_dir, f"metrics_{model_name}.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Performance metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()
