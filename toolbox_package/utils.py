import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt



def plot_learning_curve(model, X, y, train_sizes):
# Get train scores (R2), train sizes, and validation scores using `learning_curve`
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=model, X=X, y=y, train_sizes=train_sizes, cv=5)

    # Take the mean of cross-validated train scores and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, label = 'Training score')
    plt.plot(train_sizes, test_scores_mean, label = 'Test score')
    plt.ylabel('r2 score', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Learning curves', fontsize = 18, y = 1.03)
    plt.legend()
