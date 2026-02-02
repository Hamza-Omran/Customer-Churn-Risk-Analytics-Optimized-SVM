import numpy as np

class SubgradientDescentSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.0005, n_iterations=1000, batch_size=32):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iterations
        self.batch_size = batch_size
        self.w = None
        self.b = None
        self.losses = []
        
    def _hinge_loss(self, margin):
        return max(0, 1 - margin)
    
    def _compute_loss(self, X, y):
        margins = y * (np.dot(X, self.w) + self.b)
        hinge_loss = np.mean([self._hinge_loss(m) for m in margins])
        reg_loss = self.lambda_param * np.sum(self.w ** 2)
        return hinge_loss + reg_loss
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        self.w = np.zeros(n_features)
        self.b = 0
        
        for iteration in range(self.n_iters):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y_[indices]
            
            for i in range(0, n_samples, self.batch_size):
                batch_X = X_shuffled[i:i+self.batch_size]
                batch_y = y_shuffled[i:i+self.batch_size]
                
                dw = 2 * self.lambda_param * self.w
                db = 0
                
                for j in range(len(batch_X)):
                    margin = batch_y[j] * (np.dot(batch_X[j], self.w) + self.b)
                    
                    if margin < 1:
                        dw -= batch_y[j] * batch_X[j]
                        db -= batch_y[j]
                
                dw /= len(batch_X)
                db /= len(batch_X)
                
                lr_t = self.lr / (1 + 0.01 * iteration)
                self.w -= lr_t * dw
                self.b -= lr_t * db
            
            if iteration % 10 == 0:
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
