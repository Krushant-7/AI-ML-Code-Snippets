# AI-ML-Code-Snippets

Essential Python code snippets for AI/ML beginners - NLP, ML algorithms, deep learning, and TensorFlow examples.

## 🚀 Quick Start

```bash
git clone https://github.com/Krushant-7/AI-ML-Code-Snippets.git
cd AI-ML-Code-Snippets
```

## 📚 Snippets Included

### 1. **Text Preprocessing & NLP**

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required data
nltk.download('punkt')
nltk.download('stopwords')

text = "Machine Learning is amazing!"

# Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered = [word for word in tokens if word.lower() not in stop_words]
print("Filtered:", filtered)

# Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in filtered]
print("Stemmed:", stemmed)
```

### 2. **Train-Test Split & Scaling**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample data
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 0, 1])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
```

### 3. **Simple Linear Regression**

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Generate sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Evaluate
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print(f"R² Score: {r2:.4f}")
print(f"MSE: {mse:.4f}")
```

### 4. **Decision Tree Classifier**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

accuracy = dt.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

### 5. **Neural Network with TensorFlow**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create simple neural network
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())
```

## 🎨 Features

- ✅ Beginner-friendly code snippets
- 📚 Well-commented examples
- 🎉 Real-world use cases
- 🚜 Best practices included
- 🚀 Easy to copy & paste

## 📄 Topics Covered

- NLP & Text Processing
- Machine Learning Algorithms
- Data Preprocessing
- Neural Networks
- TensorFlow/Keras
- Model Evaluation
- Scikit-learn utilities

## 📝 Prerequisites

```bash
pip install numpy pandas scikit-learn tensorflow nltk
```

## 🤝 Contributing

Found a bug or have a suggestion? Feel free to open an issue or submit a PR!

## 📄 License

MIT License - feel free to use these snippets in your projects!

## 👋 Connect

- **GitHub**: [Krushant-7](https://github.com/Krushant-7)
- **LinkedIn**: [Krushant Pachauri](https://linkedin.com/in/krushant-pachauri)
- **Twitter**: [@Krushant2007](https://twitter.com/Krushant2007)

---

**Made with ❤️ by Krushant Pachauri | AI/ML Enthusiast | JEE Aspirant**
