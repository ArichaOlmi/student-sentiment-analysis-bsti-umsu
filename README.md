# Student Sentiment Analysis for BSTI UMSU Using TF-IDF and Multinomial Naïve Bayes

A text-based sentiment analysis system developed to evaluate student satisfaction toward **Information Systems and Information Technology (IS/IT)** services at **Universitas Muhammadiyah Sumatera Utara (UMSU)** using **TF-IDF feature extraction** and the **Multinomial Naïve Bayes** classification algorithm.

This project supports institutional service evaluation through automated analysis of student-generated textual feedback.

---

## Research Overview

The increasing reliance on Information Systems and Technology (IS/IT) services in higher education institutions highlights the need for objective and scalable service evaluation approaches. Traditional manual analysis of open-ended student feedback is often inefficient and difficult to interpret consistently.

This research proposes a **text-based sentiment classification framework** to analyze student perceptions regarding BSTI UMSU services.

The system classifies student comments into three sentiment categories:

- Positive
- Neutral
- Negative

The proposed framework integrates:

- Text preprocessing
- TF-IDF feature extraction
- Multinomial Naïve Bayes classification
- Manual sentiment labeling
- Streamlit-based visualization dashboard

---

## Research Objectives

This project aims to:

1. Analyze student sentiment toward BSTI UMSU services.
2. Develop an automated sentiment classification system.
3. Support institutional service evaluation through data-driven insights.
4. Provide a web-based tool for sentiment labeling and visualization.

---

## Methodology

### Data Collection

- Data source: **Google Forms questionnaire**
- Respondents: UMSU students
- Initial dataset: **620 comment documents**
- Final dataset after preprocessing: **586 non-empty documents**

### Text Preprocessing

The preprocessing pipeline includes:

- Case Folding
- Tokenization
- Stopword Removal
- Stemming (PySastrawi)

### Feature Extraction (TF-IDF)

| Parameter | Value |
|-----------|-------|
| ngram_range | (1,1) |
| min_df | 2 |
| max_df | 0.90 |
| max_features | 5000 |

### Classification Model

- Algorithm: **Multinomial Naïve Bayes**
- Evaluation Scheme: **Train-Test Split (80:20)**

---

## Experimental Results

The model achieved the following performance:

| Metric | Score |
|--------|-------|
| Accuracy | **72.03%** |
| Weighted F1-Score | **0.6870** |

### Classification Report

| Class | Precision | Recall | F1-Score |
|--------|------------|---------|-----------|
| Negative | 0.8846 | 0.5610 | 0.6866 |
| Neutral | 1.0000 | 0.1429 | 0.2500 |
| Positive | 0.6667 | 0.9524 | 0.7843 |

The findings indicate that **positive sentiment was recognized most effectively**, whereas **neutral sentiment remained the most challenging category due to semantic ambiguity**.

---

## System Features

The web-based system provides:

✅ Manual Sentiment Labeling Module  
✅ Text Preprocessing Pipeline  
✅ TF-IDF Feature Representation  
✅ Multinomial Naïve Bayes Classification  
✅ Sentiment Visualization Dashboard  
✅ Service Evaluation Report Export

---

## Running the Application

### 1. Open the Project Folder in VS Code

Open **Visual Studio Code (VS Code)** → **File** → **Open Folder...**

Select the project folder:

```text
D:\Kuliah\Skripsi\sentimen-umsu
```

### 2. Open a New Terminal

Navigate to:

```text
Terminal → New Terminal
```

### 3. Activate the Virtual Environment

```bash
.venv\Scripts\activate
```

### 4. Run the Streamlit Application

```bash
streamlit run Home.py --server.port 8502
```

### 5. Open the Application in Browser

Open the Local URL displayed in the terminal, for example:

```text
http://localhost:8502
```

---

## Technologies Used

- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy
- PySastrawi
- TF-IDF
- Multinomial Naïve Bayes

---

## Project Structure

```bash
sentimen-umsu/
│── Home.py
│── requirements.txt
│── README.md
│── pages/
│── dataset/
│── model/
│── assets/
│── .streamlit/
```

---

## Author

### Aricha Olmi Hasibuan

Information Systems Study Program  
Faculty of Computer Science and Information Technology  
Universitas Muhammadiyah Sumatera Utara  
Medan, Indonesia

📧 **Email:** arichahsb0804@gmail.com

### Co-Author

**Yoshida Sary**

Department of Educational Research and Evaluation  
Graduate School  
Universitas Negeri Yogyakarta  
Yogyakarta, Indonesia

---

## Academic Purpose

This repository was developed for **academic research purposes** related to student sentiment analysis and BSTI UMSU service evaluation.

---

## Citation

If you use this project or methodology in academic work, please cite:

**Hasibuan, A. O., & Sary, Y.**  
*Student Sentiment Analysis for BSTI UMSU Using TF-IDF and Multinomial Naïve Bayes.*
