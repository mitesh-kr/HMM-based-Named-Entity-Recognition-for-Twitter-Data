# HMM-based Named Entity Recognition for Twitter Data

This repository contains an implementation of Hidden Markov Models (HMM) for Named Entity Recognition in Twitter data. The project uses various configurations of HMMs, including bigram and trigram models with and without contextual emission probabilities.

## Task

1. Identify all named entities (binary classification)
2. Identify fine-grained named entity types (10 classes: person, product, company, geolocation, movie, music artist, tvshow, facility, sports team, others)

## Dataset

The dataset contains Twitter data with token-level annotations:
- Train.txt: Training data with word-tag pairs
- Valid.txt: Validation data
- Test.txt: Test data for final evaluation

Format: Each line contains `<Word \t Tag>`, with sentences separated by blank lines.

## Model

The implementation uses Hidden Markov Models with the following configurations:
- Bigram model without context
- Bigram model with context
- Trigram model without context
- Trigram model with context

Where "context" refers to using the previous tag when calculating emission probabilities.

## Installation

```bash
git clone https://github.com/yourusername/hmm-ner-twitter.git
cd hmm-ner-twitter
pip install -r requirements.txt
```

## Usage

### Training

```bash
python scripts/train.py --config configs/model_configs.json
```

### Evaluation

```bash
python scripts/evaluate.py --model_path results/model_bigram_with_context.pkl --test_file data/test.txt
```

### Prediction

```bash
python scripts/predict.py --model_path results/model_bigram_with_context.pkl --input_file data/test.txt --output_file results/predictions.txt
```

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Bigram without context | X.XX | X.XX | X.XX | X.XX |
| Bigram with context | X.XX | X.XX | X.XX | X.XX |
| Trigram without context | X.XX | X.XX | X.XX | X.XX |
| Trigram with context | X.XX | X.XX | X.XX | X.XX |

More detailed analysis can be found in the [report](report/NER_HMM_Report.md).

## Repository Structure

```
hmm-ner-twitter/
├── README.md
├── requirements.txt
├── src/                # Source code
├── scripts/            # Training and evaluation scripts
├── configs/            # Configuration files
├── notebooks/          # Jupyter notebooks for analysis
├── data/               # Dataset (not included in repository)
└── report/             # Project report and analysis
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
