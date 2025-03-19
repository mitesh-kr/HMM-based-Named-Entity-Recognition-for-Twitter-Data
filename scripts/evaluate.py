#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for HMM-based Named Entity Recognition on Twitter data.
"""

import os
import sys
import argparse
import json
import pickle
import time
import pandas as pd

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hmm_tagger import HMMTagger
from src.data_utils import preprocess_data
from src.evaluation import calculate_metrics, generate_classification_report
from src.visualization import plot_results_comparison


def main():
    """Main function to evaluate the HMM tagger models."""
    parser = argparse.ArgumentParser(description='Evaluate HMM-based NER models')
    parser.add_argument('--models_dir', type=str, default='models',
                        help='Directory containing the trained models')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the data files')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save the evaluation results')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Define file paths
    test_file = os.path.join(args.data_dir, 'test.txt')
    
    # Get all model files
    model_files = [f for f in os.listdir(args.models_dir) if f.startswith('model_') and f.endswith('.pkl')]
    
    if not model_files:
        print("No model files found. Please train models first.")
        sys.exit(1)
    
    # Process test data once
    print("Processing test data...")
    test_data = preprocess_data(test_file)
    test_words = [[word for word, _ in sentence] for sentence in test_data]
    test_tags = [[tag for _, tag in sentence] for sentence in test_data]
    
    # Flatten the test tags for overall evaluation
    flat_actual_tags = [tag for sent in test_tags for tag in sent]
    
    results = {}
    predictions = {}
    
    # Evaluate each model
    for model_file in model_files:
        model_name = model_file.replace('model_', '').replace('.pkl', '')
        print(f"\n{'='*50}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*50}")
        
        # Load model
        model_path = os.path.join(args.models_dir, model_file)
        model = HMMTagger()
        model.load_model(model_path)
        
        # Make predictions
        print("Making predictions...")
        start_time = time.time()
        predicted_tags = []
        
        for i, sentence in enumerate(test_words):
            if (i + 1) % 100 == 0:
                print(f"Processing sentence {i+1}/{len(test_words)}")
            predicted_tags.append(model.viterbi_decode(sentence))
        
        # Flatten predictions
        flat_predicted_tags = [tag for sent in predicted_tags for tag in sent]
        
        prediction_time = time.time() - start_time
        print(f"Prediction completed in {prediction_time:.2f} seconds")
        
        # Save predictions
        pred_file = os.path.join(args.output_dir, f"predictions_{model_name}.txt")
        with open(pred_file, 'w', encoding='utf-8') as f:
            for tags in predicted_tags:
                for tag in tags:
                    f.write(f"{tag}\n")
                f.write("\n")
        print(f"Predictions saved to {pred_file}")
        
        # Calculate metrics
        metrics = calculate_metrics(flat_actual_tags, flat_predicted_tags)
        
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test Precision: {metrics['precision']:.4f}")
        print(f"Test Recall: {metrics['recall']:.4f}")
        print(f"Test F1 Score: {metrics['f1']:.4f}")
        
        # Generate detailed classification report
        report = generate_classification_report(flat_actual_tags, flat_predicted_tags)
        print("\nDetailed Classification Report:")
        print(report)
        
        # Store results for comparison
        results[model_name] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'prediction_time': prediction_time
        }
        
        predictions[model_name] = flat_predicted_tags
        
        # Save detailed metrics
        metrics_file = os.path.join(args.output_dir, f"test_metrics_{model_name}.json")
        with open(metrics_file, 'w') as f:
            metrics_dict = {
                'model': model_name,
                'metrics': metrics,
                'prediction_time': prediction_time
            }
            json.dump(metrics_dict, f, indent=4)
        
        # Save classification report
        report_file = os.path.join(args.output_dir, f"classification_report_{model_name}.txt")
        with open(report_file, 'w') as f:
            f.write(report)
    
    # Compare results
    print("\n\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    
    df_results = pd.DataFrame(results).T
    print(df_results)
    
    # Save comparison to CSV
    csv_file = os.path.join(args.output_dir, "model_comparison.csv")
    df_results.to_csv(csv_file)
    print(f"Comparison saved to {csv_file}")
    
    # Plot results comparison
    plots_dir = os.path.join(args.output_dir, 'plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    plot_results_comparison(results, plots_dir)


if __name__ == "__main__":
    main()
