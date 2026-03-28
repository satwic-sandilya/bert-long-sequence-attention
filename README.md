# Enhanced Long Sequence Handling in BERT for Sentiment Analysis

This project addresses one of the major limitations of **BERT (`bert-base-uncased`)** — its maximum input length of **512 tokens**.

To overcome this issue for long movie reviews, a **sliding window chunking approach** was implemented along with multiple chunk aggregation strategies.

The project performs **sentiment analysis on long IMDb movie reviews** using:

- Mean Pooling
- Max Pooling
- **Attention-Based Pooling (Proposed Method)**

The attention-based pooling mechanism is the unique contribution of this work and achieved the best performance among all tested methods.

---

## Problem Statement
Standard BERT models can process only **512 tokens** at a time.

For long documents, direct truncation may remove important contextual and sentiment-rich information.

This project solves this problem using:

- **Sliding Window Tokenization**
- **Chunk-Based Aggregation**
- **Attention-Based Pooling**

---

## Dataset
Dataset used: **IMDb 50K Movie Reviews Dataset**

Source: Kaggle IMDb Dataset

The dataset contains:

- 50,000 movie reviews
- Positive and negative sentiment labels

For experimentation, a subset of **2000 samples** was used.

Split:
- Training: 1600
- Testing: 400

---

## Methodology
### 1. Sliding Window Approach
Long reviews are split into overlapping chunks of maximum **512 tokens**.

Example:

Chunk 1 → tokens 1–512  
Chunk 2 → tokens 383–894  
Chunk 3 → tokens 766–1278  

This preserves contextual continuity.

---

### 2. Pooling Methods
### Mean Pooling
Averages all chunk embeddings.

### Max Pooling
Selects the strongest feature values across chunks.

### Attention Pooling (Proposed)
Learns adaptive importance weights for each chunk.

This helps the model focus more on sentiment-rich parts of the review.

---

## Results
| Method | Accuracy | F1 Score | Training Time |
|---|---:|---:|---:|
| Mean Pooling | 0.805 | 0.8116 | 259.26 sec |
| Max Pooling | 0.805 | 0.8169 | 246.99 sec |
| Attention Pooling | **0.835** | **0.8421** | **236.53 sec** |

Final larger-data evaluation using attention pooling:

- Accuracy: **0.7825**
- F1 Score: **0.7629**
- Time: **1009.4 sec**

---

## Technologies Used
- Python
- PyTorch
- Transformers
- Pandas
- Scikit-learn
- Jupyter Notebook

---

## Files
- `long_sequence_bert.ipynb` → complete implementation notebook
- `README.md` → project overview
- `report.pdf` → final assignment report

---

## Conclusion
The proposed **attention-based pooling method** outperformed mean and max pooling for long sequence sentiment analysis.

This project demonstrates an effective solution for handling long text inputs in BERT-based models.

---

## Author
Satwic Sandilya
