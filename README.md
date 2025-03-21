# HMM-based Named Entity Recognition for Twitter Data

This repository contains an implementation of Hidden Markov Models (HMM) for Named Entity Recognition in Twitter data. The project uses various configurations of HMMs, including bigram and trigram models with and without contextual emission probabilities.

## [Google colab](https://colab.research.google.com/drive/1qUiYMd67DAH-FRwhpanWl3H2iau75onE?usp=sharing)

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
git clone https://github.com/mitesh-kr/HMM-based-Named-Entity-Recognition-for-Twitter-Data.git
cd HMM-based-Named-Entity-Recognition-for-Twitter-Data
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

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Bigram (No Context) | 0.6378 | 0.8949 | 0.6378 | 0.7424 |
| Bigram (With Context) | 0.6748 | 0.8746 | 0.6748 | 0.7597 |
| Trigram (No Context) | 0.9029 | 0.8168 | 0.9029 | 0.8577 |
| Trigram (With Context) | 0.9033 | 0.8168 | 0.9033 | 0.8579 |

More detailed analysis can be found in the [report](report/NER_HMM_Report.md).

## Repository Structure

```
HMM-based-Named-Entity-Recognition-for-Twitter-Data/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── hmm_tagger.py
│   ├── data_utils.py
│   ├── evaluation.py
│   └── visualization.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── configs/
│   └── model_configs.json
├── notebooks/
│   └── model_analysis.ipynb
├── data/
│   ├── .gitignore
│   └── README.md
└── report/
    └── NER_HMM_Report.md
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
