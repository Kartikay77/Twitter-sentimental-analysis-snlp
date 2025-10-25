# SENTIMENT ANALYSIS OF TEXT EMOTION ON TWITTER

Negation-aware Twitter preprocessing + classic ML baselines.  
Cleans tweets (URLs, @mentions, #hashtags, emojis, contractions, elongated words), **preserves negations**, then trains TF-IDF / Count models (LogReg, Linear SVM, Naive Bayes, Random Forest).  
Includes ROC curve (AUC), PR curve (optional), macro-F1/accuracy, and a saved scikit-learn **Pipeline**.

---

## ✨ Highlights
- Twitter-specific cleaning with optional POS-lemmatization
- No data leakage (vectorizers fit on **train** only; transform val/test)
- Metrics: **macro-F1**, accuracy, **ROC-AUC**, confusion @ Youden’s J
- Reproducible (`random_state=42`)
- Serialized model: `twitter_sentiment_pipeline.joblib`
- Tiny CLI for inference: `predict.py`

---

## 🗂️ Dataset
- File: `text_emotion.csv`
- Columns used: `tweet_id`, `sentiment`, `content` (dropped `author`)
- Labels kept: **happiness** and **sadness** (others filtered out)

**Cleaning pipeline (short):**  
lower → URL/@ tokens → strip `rt` → keep hashtag word → `emoji.demojize` → expand contractions → compress elongations → numbers→`NUM` → strip punct/whitespace → (optional) POS-lemmatize → remove stopwords **but keep negations**.

---

## ⚙️ Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
python - <<'PY'
import nltk
for x in ['stopwords','wordnet','omw-1.4','averaged_perceptron_tagger','vader_lexicon']:
    try:
        (nltk.data.find(f'taggers/{x}') if x=='averaged_perceptron_tagger' else nltk.data.find(f'corpora/{x}'))
    except LookupError:
        nltk.download(x)
PY
```
## 🚀 Train & Evaluate
Notebook
Open and run:
'18BCE2199_NLP_PROJECT_twitter_sentimental_analysis.ipynb'


Script (end-to-end)
'python 18bce2199_nlp_project_twitter_sentimental_analysis.py'


This trains with GridSearchCV, prints metrics, plots curves, and saves:
'twitter_sentiment_pipeline.joblib'

## 📈 Results (current run)
ROC-AUC (test): 0.889
Best held-out accuracy: ~0.80 (Count n-grams + Linear SVM)

![a](https://github.com/Kartikay77/Twitter-sentimental-analysis-snlp/blob/main/SNLP1.png)
![b](https://github.com/Kartikay77/Twitter-sentimental-analysis-snlp/blob/main/SNLP2.png)

## 📈 Results (final)
| Model | Accuracy | Precision | Recall | F1-Score |
|:--|:--:|:--:|:--:|:--:|
| **SVM (Count Vector)** | 0.789 | 0.789 | 0.789 | 0.789 |
| **Ensemble (Top-4 Count Vector)** | 0.788 | 0.789 | 0.788 | 0.788 |
| **Logistic Regression (Count Vector)** | 0.784 | 0.784 | 0.784 | 0.784 |
| **Random Forest (Count Vector)** | 0.770 | 0.770 | 0.770 | 0.770 |
| **Naive Bayes (Count Vector)** | 0.768 | 0.769 | 0.768 | 0.768 |

Confusion Matrix (Ensemble):  
[[1065 238]
[ 312 979]]


**Macro F1 ≈ 0.79 / Accuracy ≈ 0.79**  
Balanced performance across both *happiness* and *sadness* classes.

**Clarify the Ensemble**
Top-4 Voting Ensemble combines SVM, Logistic Regression, Random Forest, and Naive Bayes Count-Vector models for more robust mood classification (macro F1 ≈ 0.79).

## 🔮 Inference (CLI)
After a run has produced twitter_sentiment_pipeline.joblib:

python predict.py --model twitter_sentiment_pipeline.joblib \
  --text "I am very happy today! The atmosphere looks cheerful" \
  --text "This is quite depressing. I am filled with sorrow"

Example output:
[happiness] I am very happy today! The atmosphere looks cheerful
[sadness]   This is quite depressing. I am filled with sorrow

## 🧰 Tech Stack
Python, pandas, NumPy, matplotlib, scikit-learn, NLTK, emoji, contractions, joblib

## 🧪 Baseline (optional)
Compare against VADER to show lift:
python baseline_vader.py
prints baseline accuracy and macro-F1 on happiness/sadness


## ⚖️ Notes & Ethics
Tweets can be noisy/biased; this model is for educational/demo purposes, not high-stakes decisions.


## 📜 License
Add a LICENSE file (e.g., MIT).
MARKDOWN
--- requirements.txt ---
cat > requirements.txt <<'REQ'
scikit-learn>=1.3
pandas>=1.5
numpy>=1.24
matplotlib>=3.7
nltk>=3.8
emoji==2.10.1
contractions>=0.1
joblib>=1.3
REQ




 
