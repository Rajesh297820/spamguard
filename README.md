# 🛡️ SpamGuard – AI Spam Email Classifier

SpamGuard is a professional NLP-based text spam classifier built with Python, scikit-learn and Streamlit. It predicts whether a message is likely to be **Spam** or **Not Spam**.

> **Dataset note:** The model uses the public **SMS Spam Collection** from the UCI Machine Learning Repository as the training corpus. It contains 5,574 labeled messages. Because the corpus is SMS-focused, this project should be described as an **NLP spam text classifier** or as an email-style spam classifier trained on SMS data, not as a model trained on a dedicated corporate email dataset.

## Features

- NLP text preprocessing
- TF-IDF feature extraction
- Logistic Regression classification
- Spam / Not Spam prediction
- Confidence score
- Spam vs Not Spam probability display
- Professional Streamlit interface
- Automatic UCI dataset download during training
- GitHub-ready documentation

## Project Structure

```text
SpamGuard/
│
├── app.py
├── train_model.py
├── spam_classifier.pkl       # Generated after training
├── vectorizer.pkl            # Generated after training
├── requirements.txt
├── README.md
│
├── data/
│   └── spam.csv              # Generated automatically from UCI dataset
│
└── screenshots/
```

## Technologies

- Python 3.12+
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- Joblib
- Streamlit

## Setup on Windows / VS Code

### 1. Open the project

Open the `SpamGuard` folder in VS Code.

### 2. Create a virtual environment

Open the VS Code terminal and run:

```powershell
py -3.12 -m venv .venv
```

### 3. Activate the environment

```powershell
.venv\Scripts\activate
```

After activation, the terminal should show `(.venv)`.

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Train the model

```powershell
py -3.12 train_model.py
```

The training script will:

1. Download the UCI SMS Spam Collection if `data/spam.csv` does not exist.
2. Convert it into `data/spam.csv`.
3. Clean the text.
4. Split the data into training and testing sets.
5. Train a TF-IDF + Logistic Regression model.
6. Print accuracy, precision, recall, F1 score and the confusion matrix.
7. Create `spam_classifier.pkl` and `vectorizer.pkl`.

### 6. Start the application

```powershell
streamlit run app.py
```

Your browser should open the SpamGuard web application automatically. If it does not, copy the local URL shown in the terminal into your browser.

## Example messages for testing

### Spam-style example

```text
Congratulations! You have won a cash prize. Click the link now to claim your reward.
```

### Normal example

```text
Hi, are we still meeting at the library after class?
```

## Model workflow

```text
Raw Message
    ↓
Text Cleaning
    ↓
TF-IDF Vectorization
    ↓
Logistic Regression
    ↓
Spam / Not Spam
    ↓
Confidence Score
```

## Important limitation

The training data is the UCI SMS Spam Collection, so performance on modern business email, phishing campaigns, or other domains may differ. The confidence value is a model probability estimate and should not be treated as a guarantee.

## Dataset and citation

Almeida, T. & Hidalgo, J. (2011). **SMS Spam Collection**. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5CC84

Dataset license: **CC BY 4.0**. Give appropriate credit when redistributing or adapting the dataset.

## Author

**Rajesh Behera**

NLP / Machine Learning Internship Project
