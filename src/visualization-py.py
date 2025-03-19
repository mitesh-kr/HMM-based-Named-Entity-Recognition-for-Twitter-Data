import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot confusion matrix for NER tag prediction
    
    Args:
        cm (numpy.ndarray): Confusion matrix
        classes (list): List of class names
        title (str): Title for the plot
        cmap: Color map for the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'report/confusion_matrix_{title.replace(" ", "_")}.png')
    plt.close()

def plot_tag_distribution(tags, title='Tag Distribution'):
    """
    Plot distribution of named entity tags
    
    Args:
        tags (list): List of tags
        title (str): Title for the plot
    """
    tag_counts = pd.Series(tags).value_counts()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=tag_counts.index, y=tag_counts.values)
    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel('Tag')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'report/tag_distribution_{title.replace(" ", "_")}.png')
    plt.close()

def plot_metrics_comparison(results, metric='f1_score', title='Model Comparison'):
    """
    Plot comparison of different models based on metrics
    
    Args:
        results (dict): Dictionary with model names as keys and metric values as values
        metric (str): Metric to plot (accuracy, precision, recall, f1_score)
        title (str): Title for the plot
    """
    plt.figure(figsize=(10, 6))
    models = list(results.keys())
    values = [results[model][metric] for model in models]
    
    sns.barplot(x=models, y=values)
    plt.title(f'{title} - {metric.replace("_", " ").title()}')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xlabel('Model')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(f'report/{metric}_comparison.png')
    plt.close()

def plot_learning_curve(train_scores, valid_scores, iterations, title='Learning Curve'):
    """
    Plot learning curve for model training
    
    Args:
        train_scores (list): List of training scores
        valid_scores (list): List of validation scores
        iterations (list): List of iteration numbers
        title (str): Title for the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_scores, 'o-', label='Training')
    plt.plot(iterations, valid_scores, 'o-', label='Validation')
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'report/learning_curve_{title.replace(" ", "_")}.png')
    plt.close()

def plot_transition_heatmap(transition_matrix, states, title='Transition Probabilities'):
    """
    Plot heatmap of transition probabilities
    
    Args:
        transition_matrix (numpy.ndarray): Matrix of transition probabilities
        states (list): List of state names
        title (str): Title for the plot
    """
    plt.figure(figsize=(12, 10))
    mask = transition_matrix == 0
    sns.heatmap(transition_matrix, annot=False, cmap='viridis', 
                xticklabels=states, yticklabels=states, mask=mask)
    plt.title(title)
    plt.ylabel('From State')
    plt.xlabel('To State')
    plt.tight_layout()
    plt.savefig(f'report/transition_heatmap_{title.replace(" ", "_")}.png')
    plt.close()
