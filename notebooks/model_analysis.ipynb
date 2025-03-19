{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# HMM-based NER Model Analysis\n",
    "\n",
    "This notebook provides analysis of the HMM-based Named Entity Recognition models for Twitter data. We'll examine:\n",
    "1. Data distribution\n",
    "2. Model performance comparison\n",
    "3. Error analysis\n",
    "4. Visualization of HMM parameters"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import sys\n",
    "import os\n",
    "sys.path.append('..')\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from src.data_utils import load_data, preprocess_data\n",
    "from src.hmm_tagger import HMMTagger\n",
    "from src.evaluation import evaluate_model, compute_metrics\n",
    "from src.visualization import (\n",
    "    plot_confusion_matrix, \n",
    "    plot_tag_distribution, \n",
    "    plot_metrics_comparison,\n",
    "    plot_transition_heatmap\n",
    ")\n",
    "\n",
    "import pickle\n",
    "import json\n",
    "\n",
    "%matplotlib inline\n",
    "plt.style.use('ggplot')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Data Loading and Exploration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Load data\n",
    "train_data = load_data('../data/train.txt')\n",
    "valid_data = load_data('../data/valid.txt')\n",
    "test_data = load_data('../data/test.txt')\n",
    "\n",
    "print(f\"Training sentences: {len(train_data)}\")\n",
    "print(f\"Validation sentences: {len(valid_data)}\")\n",
    "print(f\"Test sentences: {len(test_data)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Extract all tags from training data\n",
    "all_tags = [tag for sentence in train_data for _, tag in sentence]\n",
    "unique_tags = list(set(all_tags))\n",
    "print(f\"Number of unique tags: {len(unique_tags)}\")\n",
    "print(f\"Tags: {unique_tags}\")\n",
    "\n",
    "# Plot tag distribution\n",
    "plot_tag_distribution(all_tags, title='Training Data Tag Distribution')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Model Performance Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Load model results\n",
    "with open('../configs/model_results.json', 'r') as f:\n",
    "    results = json.load(f)\n",
    "\n",
    "# Compare models\n",
    "plot_metrics_comparison(results, metric='accuracy', title='Model Accuracy Comparison')\n",
    "plot_metrics_comparison(results, metric='precision', title='Model Precision Comparison')\n",
    "plot_metrics_comparison(results, metric='recall', title='Model Recall Comparison')\n",
    "plot_metrics_comparison(results, metric='f1_score', title='Model F1 Score Comparison')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Error Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Load the best model\n",
    "with open('../models/best_model.pkl', 'rb') as f:\n",
    "    best_model = pickle.load(f)\n",
    "\n",
    "# Predict on test data\n",
    "test_words = [[word for word, _ in sentence] for sentence in test_data]\n",
    "test_true = [[tag for _, tag in sentence] for sentence in test_data]\n",
    "test_pred = [best_model.viterbi_decode(sentence) for sentence in test_words]\n",
    "\n",
    "# Calculate confusion matrix\n",
    "from sklearn.metrics import confusion_matrix\n",
    "true_flat = [tag for sent in test_true for tag in sent]\n",
    "pred_flat = [tag for sent in test_pred for tag in sent]\n",
    "cm = confusion_matrix(true_flat, pred_flat, labels=unique_tags)\n",
    "\n",
    "# Plot confusion matrix\n",
    "plot_confusion_matrix(cm, unique_tags, title='Best Model Confusion Matrix')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. HMM Parameter Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Visualize start probabilities\n",
    "start_probs = pd.Series(best_model.start_prob)\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.barplot(x=start_probs.index, y=start_probs.values)\n",
    "plt.title('Start Probabilities')\n",
    "plt.ylabel('Probability')\n",
    "plt.xlabel('Tag')\n",
    "plt.xticks(rotation=45)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Visualize transition probabilities\n",
    "if hasattr(best_model, 'trans_prob'):\n",
    "    trans_matrix = np.zeros((len(unique_tags), len(unique_tags)))\n",
    "    for i, tag1 in enumerate(unique_tags):\n",
    "        for j, tag2 in enumerate(unique_tags):\n",
    "            if (tag1, tag2) in best_model.trans_prob:\n",
    "                trans_matrix[i, j] = best_model.trans_prob[(tag1, tag2)]\n",
    "    \n",
    "    plot_transition_heatmap(trans_matrix, unique_tags, title='Transition Probabilities')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Case Studies and Error Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Find examples of common errors\n",
    "error_cases = []\n",
    "for i, (sent_true, sent_pred, words) in enumerate(zip(test_true, test_pred, test_words)):\n",
    "    for j, (true_tag, pred_tag, word) in enumerate(zip(sent_true, sent_pred, words)):\n",
    "        if true_tag != pred_tag:\n",
    "            error_cases.append((word, true_tag, pred_tag))\n",
    "            if len(error_cases) >= 20:  # Limit to 20 examples\n",
    "                break\n",
    "    if len(error_cases) >= 20:\n",
    "        break\n",
    "\n",
    "# Display error cases\n",
    "error_df = pd.DataFrame(error_cases, columns=['Word', 'True Tag', 'Predicted Tag'])\n",
    "error_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Analyze most confused tag pairs\n",
    "confused_pairs = {}\n",
    "for word, true_tag, pred_tag in error_cases:\n",
    "    pair = (true_tag, pred_tag)\n",
    "    if pair not in confused_pairs:\n",
    "        confused_pairs[pair] = 0\n",
    "    confused_pairs[pair] += 1\n",
    "\n",
    "# Sort by frequency\n",
    "sorted_pairs = sorted(confused_pairs.items(), key=lambda x: x[1], reverse=True)\n",
    "for (true_tag, pred_tag), count in sorted_pairs[:10]:  # Top 10 confused pairs\n",
    "    print(f\"True: {true_tag}, Predicted: {pred_tag}, Count: {count}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Performance on Specific Named Entity Types"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Calculate metrics per tag\n",
    "from sklearn.metrics import precision_recall_fscore_support\n",
    "\n",
    "precision, recall, f1, support = precision_recall_fscore_support(true_flat, pred_flat, labels=unique_tags)\n",
    "\n",
    "# Create per-tag metrics dataframe\n",
    "tag_metrics = pd.DataFrame({\n",
    "    'Tag': unique_tags,\n",
    "    'Precision': precision,\n",
    "    'Recall': recall,\n",
    "    'F1 Score': f1,\n",
    "    'Support': support\n",
    "})\n",
    "\n",
    "# Sort by F1 score\n",
    "tag_metrics = tag_metrics.sort_values('F1 Score', ascending=False)\n",
    "tag_metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Visualize per-tag F1 scores\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.barplot(x='Tag', y='F1 Score', data=tag_metrics)\n",
    "plt.title('F1 Score by Tag')\n",
    "plt.xticks(rotation=45)\n",
    "plt.ylim(0, 1.0)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Conclusions and Recommendations"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Based on the analysis above, we can draw the following conclusions:\n",
    "\n",
    "1. **Model Performance**: The trigram model with context during emission probability calculation performs the best overall, with an F1 score of X.XX.\n",
    "\n",
    "2. **Entity Type Performance**:\n",
    "   - Best performing entity types: [List top 3 from the analysis]\n",
    "   - Worst performing entity types: [List bottom 3 from the analysis]\n",
    "\n",
    "3. **Common Errors**:\n",
    "   - [Summarize the most common error patterns observed]\n",
    "   - [Analyze why these errors might be occurring]\n",
    "\n",
    "4. **Recommendations**:\n",
    "   - [Suggest improvements to the model]\n",
    "   - [Suggest additional features that could help]\n",
    "   - [Discuss alternative approaches that might work better]"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
