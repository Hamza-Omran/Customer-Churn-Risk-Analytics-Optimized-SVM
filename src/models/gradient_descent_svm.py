import numpy as np

class GradientDescentSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iterations
        self.w = None
        self.b = None
        self.losses = []
        
    def _smoothed_hinge_loss(self, margin):
        if margin >= 1:
            return 0
        elif margin <= 0:
            return 1 - margin
        else:
            return (1 - margin) ** 2 / 2
    
    def _compute_loss(self, X, y):
        margins = y * (np.dot(X, self.w) + self.b)
        hinge_loss = np.mean([self._smoothed_hinge_loss(m) for m in margins])
        reg_loss = self.lambda_param * np.sum(self.w ** 2)
        return hinge_loss + reg_loss
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0
        
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                margin = y_[idx] * (np.dot(x_i, self.w) + self.b)
                
                if margin >= 1:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                elif margin <= 0:
                    dw = 2 * self.lambda_param * self.w - y_[idx] * x_i
                    db = -y_[idx]
                else:
                    dw = 2 * self.lambda_param * self.w - (1 - margin) * y_[idx] * x_i
                    db = -(1 - margin) * y_[idx]
                
                self.w -= self.lr * dw
                self.b -= self.lr * db
            
            loss = self._compute_loss(X, y_)
            self.losses.append(loss)
    
    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        return np.sign(linear_output)
    
    def score(self, X, y):
        predictions = self.predict(X)
        y_ = np.where(y <= 0, -1, 1)
        accuracy = np.mean(predictions == y_)
        return accuracy
