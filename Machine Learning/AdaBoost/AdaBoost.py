import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction.text import CountVectorizer

class AdaBoost:

    
    def __init__(self, m, n, k, base_estimator=None):
        self.m = m
        self.n = n
        self.k = k
        self.vocab = None
        self.alphas = []
        self.G_M = []
        self.M = None
        self.training_errors = []
        self.prediction_errors = []
        self.base_estimator = base_estimator
        
    def fit(self, X, y, M =150):
    
        
        # Clear before calling
        self.alphas = [] 
        self.training_errors = []
        self.M = M

        # Iterate over M weak classifiers
        for m in range(0, M):
            
            # Set weights for current boosting iteration
            if m == 0:
                w_i = np.ones(len(y)) * 1 / len(y)  # At m = 0, weights are all the same and equal to 1 / N
            else:
                # (d) Update w_i
                w_i = update_weights(w_i, alpha_m, y, y_pred)
            
            # (a) Fit weak classifier and predict labels
            G_m = DecisionTreeClassifier(max_depth = 1)     # Stump: Two terminal-node classification tree
            G_m.fit(X, y, sample_weight = w_i)
            y_pred = G_m.predict(X)
            
            self.G_M.append(G_m) # Save to list of weak classifiers

            # (b) Compute error
            error_m = compute_error(y, y_pred, w_i)
            self.training_errors.append(error_m)

            # (c) Compute alpha
            alpha_m = compute_alpha(error_m) 
            self.alphas.append(alpha_m)
    def preprocess_data(self, texts, labels):
        # Convert word indices back to text
        texts = [' '.join([str(word) for word in doc]) for doc in texts]

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(texts)

        word_counts = np.array(X.sum(axis=0))[0]
        words = np.array(vectorizer.get_feature_names_out())

        # Sort words based on frequency
        sorted_indices = np.argsort(word_counts)[::-1]

        # Select vocabulary excluding n most frequent and k most rare words
        self.vocab = words[sorted_indices[self.n:self.n + 3000]]

        # Filter data based on selected vocabulary
        X_filtered = X[:, np.isin(words, self.vocab)]

        return X_filtered.toarray(), labels

    def predict(self, X):

        # Initialise dataframe with weak predictions for each observation
        weak_preds = pd.DataFrame(index = range(len(X)), columns = range(self.M)) 

        # Predict class label for each weak classifier, weighted by alpha_m
        for m in range(self.M):
            y_pred_m = self.G_M[m].predict(X) * self.alphas[m]
            weak_preds.iloc[:,m] = y_pred_m

        # Calculate final predictions
        y_pred = (1 * np.sign(weak_preds.T.sum())).astype(int)

        return y_pred

    

# Compute error rate, alpha and w
def compute_error(y, y_pred, w_i):
    
    return (sum(w_i * (np.not_equal(y, y_pred)).astype(int)))/sum(w_i)

def compute_alpha(error):
   
    return np.log((1 - error) / error)

def update_weights(w_i, alpha, y, y_pred):
    
    return w_i * np.exp(alpha * (np.not_equal(y, y_pred)).astype(int))