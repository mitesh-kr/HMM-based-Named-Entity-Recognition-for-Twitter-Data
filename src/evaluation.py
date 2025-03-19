import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def evaluate_predictions(predicted_tags, actual_tags):
    """
    Evaluate predictions against actual tags.

    Args:
        predicted_tags (list): List of predicted tags
        actual_tags (list): List of actual tags

    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(actual_tags, predicted_tags)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_tags, predicted_tags, average='weighted')

    # Generate detailed classification report
    report = classification_report(actual_tags, predicted_tags)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'report': report
    }

    return metrics


def print_evaluation_metrics(metrics, model_name=""):
    """
    Print evaluation metrics in a readable format.

    Args:
        metrics (dict): Dictionary containing evaluation metrics
        model_name (str): Name of the model for display
    """
    title = f"Evaluation Results for {model_name}" if model_name else "Evaluation Results"
    
    print(f"\n{'='*50}")
    print(title)
    print(f"{'='*50}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("\nDetailed Classification Report:")
    print(metrics['report'])


def compare_models(results):
    """
    Compare multiple models based on their evaluation metrics.

    Args:
        results (dict): Dictionary mapping model names to their evaluation metrics

    Returns:
        pandas.DataFrame: DataFrame with comparison results
    """
    df_results = pd.DataFrame()
    
    for model_name, metrics in results.items():
        df_results[model_name] = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1']
        ]
    
    df_results.index = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    return df_results


def save_predictions(predictions, filepath):
    """
    Save predictions to a file.

    Args:
        predictions (list): List of predicted tags
        filepath (str): Path to the output file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for tag in predictions:
            f.write(f"{tag}\n")
    print(f"Predictions saved to {filepath}")


def save_comparison_results(df_results, filepath):
    """
    Save model comparison results to a CSV file.

    Args:
        df_results (pandas.DataFrame): DataFrame with comparison results
        filepath (str): Path to the output CSV file
    """
    df_results.to_csv(filepath)
    print(f"Comparison results saved to {filepath}")
