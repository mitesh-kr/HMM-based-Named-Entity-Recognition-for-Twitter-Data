"""
HMM-based NER for Twitter data

This module implements Hidden Markov Models for Named Entity Recognition
on Twitter data, supporting both bigram and trigram models.
"""

from .hmm_tagger import HMMTagger
from .data_utils import load_data, preprocess_data
from .evaluation import evaluate_model, compute_metrics
from .visualization import (plot_confusion_matrix, plot_tag_distribution, 
                           plot_metrics_comparison, plot_learning_curve,
                           plot_transition_heatmap)

__all__ = [
    'HMMTagger',
    'load_data',
    'preprocess_data',
    'evaluate_model',
    'compute_metrics',
    'plot_confusion_matrix',
    'plot_tag_distribution',
    'plot_metrics_comparison',
    'plot_learning_curve',
    'plot_transition_heatmap'
]
