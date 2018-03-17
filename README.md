# Machine-Learning-Algorithms
Building machine learning algorithms from scratch (Python 3.6)

TODO:
- [x] Linear regression
- [x] Logistic regression
- [ ] Neural Network
- [ ] Support Vector Machines
## Linear Regression
Given a set of features represented as vector X, the optimised weight vector Θ (with bias included) is found by performing batch gradient descent. Consider the following case with only one feature:

The algorithm is confirmed to be working properly as the loss converges wrt to the number of iteration.

![alt text][loss_vs_iter_1]

with 1,000,000 iterations, learning rate of 0.0001 and regularisation strength of 0, the resulting Θ is the same as the one computed using LinearRegression from sklearn.linear_model.


![alt text][comparison_1]

Lastly consider the below case using 2 features; the results is still consistent with each other.

![alt text][comparison_2]


[loss_vs_iter_1]: https://github.com/bangbangjim/Machine-Learning-Algorithms/blob/master/images/loss%20vs%20iter%20LR.png "loss vs number of iteration"

[comparison_1]: https://github.com/bangbangjim/Machine-Learning-Algorithms/blob/master/images/comparison%20sklearn.png "Comparison of sklearn LinearRegression and self-built gradient descent algorithm"

[comparison_2]: https://github.com/bangbangjim/Machine-Learning-Algorithms/blob/master/images/comparison%202%20sklearn.png "Comparison (2 features)"

## Logistic Regression
Please see the Jupyter Notebook for details.
