# Sentiment-Analysis-NLP


# Sentiment Analysis using NLP Techniques

This project applies Natural Language Processing (NLP) methods on Amazon product reviews to analyze and classify sentiments. It combines rule-based (VADER) and transformer-based (RoBERTa) approaches, along with thorough exploratory data analysis and impactful visualizations.

---

## Project Objective

To perform sentiment analysis on customer reviews using both classical NLP techniques and state-of-the-art deep learning models. The aim is to identify and compare sentiment trends using both approaches and visualize their relationship with review ratings.

---

## Dataset Description

- **Dataset**: Amazon Fine Food Reviews Dataset
- **File Used**: `Reviews.csv`
- **Subset**: First 500 rows (for computational efficiency)
- **Fields**:
  - `Id`
  - `ProductId`
  - `UserId`
  - `ProfileName`
  - `HelpfulnessNumerator`
  - `HelpfulnessDenominator`
  - `Score` (Review stars)
  - `Time`
  - `Summary`
  - `Text` (Main review content)

---

## Technologies and Libraries

| Category             | Tools & Libraries                         |
|----------------------|-------------------------------------------|
| Data Handling        | `pandas`, `numpy`                         |
| Visualization        | `matplotlib`, `seaborn`                   |
| NLP (Classical)      | `nltk`, `VADER`, `POS Tagging`, `NER`     |
| NLP (Transformer)    | `transformers`, `RoBERTa`                 |
| Model Evaluation     | `scikit-learn`, `classification_report`   |
| Others               | `tqdm`, `plotly` (optional)               |

---

##  Methodology

### 1. Exploratory Data Analysis (EDA)
- Visualized review count per rating
- Investigated common patterns in review text

### 2. Classical NLP (NLTK + VADER)
- Tokenization, POS Tagging, Named Entity Recognition (NER)
- Sentiment scoring using VADER (compound score)

### 3. Deep Learning Approach (RoBERTa)
- Used pre-trained `roberta-base` from Hugging Face Transformers
- Encoded review text and classified it as positive, negative, or neutral

### 4. Merging Results
- Combined VADER and RoBERTa outputs for comparison
- Created new columns for each model’s sentiment score

### 5. Visualizations
- Bar plots for rating distribution
- Pie charts and histograms of sentiment categories
- Pair plots to compare VADER and RoBERTa sentiment scores

---

##  Key Findings

- **VADER** performs well on short, explicit text.
- **RoBERTa** handles complex and nuanced reviews more accurately.
- Positive correlation observed between compound sentiment scores and actual review ratings.
- RoBERTa yields better accuracy and generalization than VADER on textual sentiment classification.

---

## 📊 Visual Insights

- **Bar Plot**: Count of reviews by rating
- **Pie Chart**: Sentiment distribution (Positive, Negative, Neutral)
- **Scatter Plot**: VADER vs. RoBERTa scores
- **Pair Plot**: Distribution and correlation of sentiment scores

---


## How to Run

```bash
# Clone this repository
git clone https://github.com/<your-username>/Sentiment-Analysis-NLP.git
cd Sentiment-Analysis-NLP

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Sentiment_Analysis_NLP.ipynb
```

---

## Future Enhancements

- Fine-tune RoBERTa on domain-specific review data
- Incorporate BERT + LSTM hybrid models for context-aware sentiment detection
- Build a web interface for real-time review sentiment classification

---


##  Acknowledgements

- [NLTK - Natural Language Toolkit](https://www.nltk.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Amazon Fine Food Reviews - Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

---

##  Author

**Pakhi Singhal**  
AI & ML Enthusiast 

---

##  License

This project is open-source and available under the [MIT License](LICENSE).
