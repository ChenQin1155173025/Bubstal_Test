# Bubstal_Test

## Submission Overview

This submission implements two core components of an e-commerce AI assistant:

1. **Naive Bayes Sentiment Classifier** - Determines if customer sentiment is "Happy" or "Frustrated" from text input
2. **Gradient Descent Optimizer** - Minimizes a quadratic cost function using iterative optimization

Both components are implemented in **pure Python** with no external libraries.

---

## Setup

No installation or dependencies required. This runs with standard Python 3.x.

### Requirements
- Python 3.6 or higher
- No pip install needed

### Running the Code

```bash
# Run the sentiment classifier
python naive_bayes_classifier.py

# Run the optimizer
python optimizer.py

# Run all unit tests
python tests/test_all.py
```

## File Structure
submission/

├── naive_bayes_classifier.py # Task 1: Naive Bayes classifier

├── optimizer.py # Task 2: Gradient descent optimizer

├── dataset.py # Training and test data

├── tests/

│ └── test_all.py # Unit tests (7 tests)

└── README.md # This file

## Complexity Analysis

### Task 1: Naive Bayes Sentiment Classifier

train()	O(N × T)	N = number of documents, T = average tokens per document. Processes each token once.

predict()	O(T)	T = number of tokens in input. Constant factor of 2 for two classes.

For the provided dataset: N=20, T≈10 → training ~200 operations, prediction ~10 operations per sentence.

### Task 2: Gradient Descent Optimizer

optimize()	O(k)	k = number of iterations until convergence. Each iteration does O(1) work.

Typical k = 50-200 with learning_rate=0.1, tolerance=1e-6.


## Unit Tests
7 tests covering probability calculations, gradient updates, and convergence.


