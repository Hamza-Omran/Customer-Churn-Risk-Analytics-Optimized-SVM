import numpy as np
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def get_feature_importance(model, X, y, feature_names):
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    elif hasattr(model, 'w'):
        importance = np.abs(model.w)
    else:
        return None
    
    indices = np.argsort(importance)[::-1]
    
    return {
        'importance': importance,
        'indices': indices,
        'feature_names': feature_names
    }

def plot_feature_importance(importance_dict, top_n=10):
    importance = importance_dict['importance']
    indices = importance_dict['indices']
    feature_names = importance_dict['feature_names']
    
    plt.figure(figsize=(10, 6))
    plt.title('Top Feature Importances')
    top_indices = indices[:top_n]
    plt.barh(range(top_n), importance[top_indices])
    plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    return plt

def get_permutation_importance(model, X, y, feature_names):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    
    indices = np.argsort(result.importances_mean)[::-1]
    
    return {
        'importance': result.importances_mean,
        'std': result.importances_std,
        'indices': indices,
        'feature_names': feature_names
    }
