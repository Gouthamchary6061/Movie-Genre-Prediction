# Movie-Genre-Prediction

## **Project Title**
Movie Genre Prediction

## **Project Aim**

Predict the **genre of a movie** based on its **plot summary** using **text classification** techniques.

---
## **Project Description**

This project applies **Natural Language Processing (NLP)** techniques for **text classification** on a movie dataset.

* Movie plot summaries are **preprocessed** using tokenization, stopword removal, and lemmatization.
* **TF-IDF vectorization** converts the text into numerical features.
* A **Naive Bayes classifier** is trained to predict the movie genre.
* The project also demonstrates prediction of new movie plots.
---
## **Technologies Used**
* **Python**
* **Pandas & NumPy** (data handling)
* **NLTK / SpaCy** (text preprocessing)
* **Scikit-learn** (TF-IDF, Naive Bayes, train/test split)
* **Matplotlib / Seaborn** (optional for visualizations)

---

## **Project Workflow**

1. **Data Collection**

   * Dataset with `plot_summary` and `genre`.
   * Can use IMDB or Kaggle movie datasets.

2. **Data Preprocessing**

   * Remove punctuation and HTML tags.
   * Convert text to lowercase.
   * Remove stopwords and apply **lemmatization**.

3. **Feature Extraction**

   * Convert cleaned plot summaries into **TF-IDF vectors**.

4. **Model Training & Evaluation**

   * Split data into training and test sets.
   * Train a **Naive Bayes classifier**.
   * Evaluate using **accuracy and classification report**.

5. **Prediction**

   * Predict the genre of new movie plots.

---

## **How to Run**

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
```

2. Install required libraries:

```bash
pip install pandas numpy scikit-learn nltk
```

3. Run the Python script:

```bash
python movie_genre_prediction.py
```

---

## **Sample Output**

* Accuracy and classification report of the model.
* Predicted genres for new movie plot examples:

```
Plot: "A young man discovers his magical powers while attending a school for wizards."
Predicted Genre: Fantasy

Plot: "A detective investigates a series of mysterious murders in the city."
Predicted Genre: Thriller
```
