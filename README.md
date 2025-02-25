# Movie Review Sentiment Classifier

This project is a **movie review sentiment classifier** using **Logistic Regression**. It processes a dataset of movie reviews, vectorizes the text, trains a model, and predicts whether a review is **positive** or **negative**.

## Features
- Reads and cleans movie reviews from `dataset.csv`.
- Converts text into numerical vectors using `CountVectorizer`.
- Trains a **Logistic Regression** model and evaluates accuracy.
- Saves the trained model and vectorizer using `pickle`.
- Allows users to input a review and predict its sentiment.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/movie-review-classifier.git
   cd movie-review-classifier


2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Ensure you have a dataset (`dataset.csv`) with the following columns:
   - `review`: The movie review text.
   - `sentiment`: Label ("positive" or "negative").

## Usage

### Train the Model
Run the following command to train the model and save it:
```sh
python train.py
```

### Predict a Review’s Sentiment
Once trained, you can classify a new review:
```sh
python predict.py
```
The program will ask for a review and return whether it is **positive** or **negative**.

## File Structure
```
movie-review-classifier/
│── dataset.csv            # Movie reviews dataset
│── train.py               # Script to train the model
│── test.py             # Script to classify a new review
│── model.model            # Saved Logistic Regression model
│── vectorizer.model       # Saved CountVectorizer
│── README.md              # Project documentation
│── requirements.txt       # Python dependencies
```

## Dependencies
- `scikit-learn`
- `pickle`
- `csv`
- `re`
