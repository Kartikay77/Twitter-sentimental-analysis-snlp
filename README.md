cat > README.md <<'EOF'
# SENTIMENT ANALYSIS OF TEXT EMOTION ON TWITTER

Negation-aware Twitter preprocessing + classic ML baselines for **happiness vs sadness** classification.

🔗 **Live Demo:** https://twitter-sentimental-analysis-snlp-9bjhayncgxvsyjif5aybqy.streamlit.app

This project cleans noisy Twitter text (URLs, @mentions, hashtags, emojis, contractions, elongated words), **preserves negations**, and trains multiple classic ML models using TF-IDF / Count features (LogReg, Linear SVM, Naive Bayes, Random Forest). It also includes a deployed **Streamlit app** for real-time sentiment prediction.

---

## ✨ Highlights
- Twitter-specific preprocessing with optional POS-aware lemmatization
- Preserves negation words (e.g., *not, never, no*) during stopword removal
- No data leakage (vectorizers fit on **train** only; val/test use `transform`)
- Metrics: **macro-F1**, accuracy, ROC-AUC, confusion matrix
- Reproducible training (`random_state=42`)
- Saved scikit-learn pipeline artifact for inference
- Streamlit web app for interactive predictions

---

## 🗂️ Dataset
- **File:** `text_emotion.csv`
- **Columns used:** `tweet_id`, `sentiment`, `content` (`author` dropped)
- **Labels kept:** `happiness`, `sadness` (other labels filtered out)

### Cleaning pipeline (summary)
Lowercasing → URL/@mention normalization → remove `RT` → keep hashtag word → emoji text conversion → expand contractions → normalize elongated words → normalize numbers → strip punctuation/extra spaces → (optional) POS lemmatization → remove stopwords **while keeping negations**.

---

## ⚙️ Setup
```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```
## NLTK resources (one-time download)
python - <<'PY'
import nltk

resources = [
    'stopwords',
    'wordnet',
    'omw-1.4',
    'averaged_perceptron_tagger',
    'vader_lexicon'
]

for r in resources:
    try:
        if r == 'averaged_perceptron_tagger':
            nltk.data.find(f'taggers/{r}')
        elif r == 'vader_lexicon':
            nltk.data.find(f'sentiment/{r}')
        else:
            nltk.data.find(f'corpora/{r}')
    except LookupError:
        nltk.download(r)
PY

--
## 🚀 Train & Evaluate
## Notebook
## Run:
```
18BCE2199_NLP_PROJECT_twitter_sentimental_analysis.ipynb
```
```
python 18bce2199_nlp_project_twitter_sentimental_analysis.py
```
This trains with GridSearchCV, prints metrics, plots curves, and saves the trained pipeline artifact:
```
twitter_sentiment_model_artifacts.joblib
```

## 🌐 Streamlit App (Live Demo)
The deployed Streamlit app loads the trained scikit-learn pipeline and performs real-time sentiment inference on user-entered text.
Live App: https://twitter-sentimental-analysis-snlp-9bjhayncgxvsyjif5aybqy.streamlit.app

## 📈 Results (current run)
ROC-AUC (test): 0.889
Best held-out accuracy: ~0.80 (Count n-grams + Linear SVM)
