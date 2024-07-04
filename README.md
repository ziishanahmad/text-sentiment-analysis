
```markdown
# Text Sentiment Analysis

## Overview
This project aims to develop a machine learning model to perform sentiment analysis on IMDb movie reviews. The dataset contains 50,000 movie reviews, split evenly into 25,000 for training and 25,000 for testing. The goal is to classify each review as either positive or negative.

## Key Features
- **Data Preprocessing:** Loading and padding the IMDb dataset.
- **Model Building:** Training a Long Short-Term Memory (LSTM) network to classify sentiment.
- **Model Evaluation:** Assessing model performance using accuracy, loss, classification report, and confusion matrix.
- **Model Visualization:** Visualizing training history and confusion matrix.
- **Sentiment Prediction:** Predicting sentiment for new sentences.

## Installation

### Clone the Repository
To get started, clone this repository to your local machine using the following command:
```sh
git clone https://github.com/ziishanahmad/text-sentiment-analysis.git
cd text-sentiment-analysis
```

### Set Up a Virtual Environment
It is recommended to use a virtual environment to manage your dependencies. You can set up a virtual environment using `venv`:
```sh
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Required Libraries
Install the necessary libraries using `pip`:
```sh
pip install -r requirements.txt
```

## Usage

### Run the Jupyter Notebook
Open the Jupyter notebook to run the project step-by-step:
1. Launch Jupyter Notebook:
   ```sh
   jupyter notebook
   ```
2. Open the `text_sentiment_analysis.ipynb` notebook.
3. Run the cells step-by-step to preprocess the data, train the model, evaluate its performance, and visualize the results.

## Detailed Explanation of the Code

### Import Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
```

### Load the Dataset
```python
max_features = 10000
maxlen = 100
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
```

### Build the Model
```python
model = Sequential([
    Embedding(max_features, 128, input_length=maxlen),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Train the Model
```python
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
```

### Evaluate the Model
```python
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

### Predict the Sentiment of a Specific Sentence
```python
word_index = imdb.get_word_index()

def preprocess_sentence(sentence, word_index, maxlen):
    tokenizer = Tokenizer(num_words=max_features)
    words = sentence.lower().split()
    sequence = [word_index.get(word, 0) for word in words]
    padded_sequence = pad_sequences([sequence], maxlen=maxlen)
    return padded_sequence

test_sentence = "This movie was fantastic! The performances were brilliant."
test_padded = preprocess_sentence(test_sentence, word_index, maxlen)

prediction = model.predict(test_padded)
sentiment = 'positive' if prediction > 0.5 else 'negative'
print(f'The sentiment of the test sentence is: {sentiment}')
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License.

## Acknowledgements
- The IMDb dataset is provided by IMDb.
- The developers of TensorFlow for their deep learning framework.

## Contact
For any questions or feedback, please contact:
- **Name:** Zeeshan Ahmad
- **Email:** ziishanahmad@gmail.com
- **GitHub:** [ziishanahmad](https://github.com/ziishanahmad)
- **LinkedIn:** [ziishanahmad](https://www.linkedin.com/in/ziishanahmad/)
```
