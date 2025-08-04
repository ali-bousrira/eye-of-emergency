import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import unique_labels
from scipy.sparse import issparse

class CustomDecisionTree(BaseEstimator, ClassifierMixin):
    """
    Implémentation personnalisée d'un arbre de décision pour la classification binaire
    Compatible avec scikit-learn pour utilisation dans Pipeline et GridSearchCV
    """
    
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.classes_ = None
        self.n_classes_ = None
        self.n_features_in_ = None
    
    def _entropy(self, y):
        """Calcule l'entropie d'un ensemble de labels"""
        if len(y) == 0:
            return 0
        
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        return entropy
    
    def _information_gain(self, X, y, feature_idx, threshold):
        """Calcule le gain d'information pour une division donnée"""
        # Division des données
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return 0
        
        # Calcul du gain d'information
        n = len(y)
        n_left, n_right = np.sum(left_mask), np.sum(right_mask)
        
        entropy_parent = self._entropy(y)
        entropy_left = self._entropy(y[left_mask])
        entropy_right = self._entropy(y[right_mask])
        
        weighted_entropy = (n_left/n) * entropy_left + (n_right/n) * entropy_right
        information_gain = entropy_parent - weighted_entropy
        
        return information_gain
    
    def _best_split(self, X, y):
        """Trouve la meilleure division pour un nœud"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_idx, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Construit récursivement l'arbre de décision"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # Conditions d'arrêt
        if (depth >= self.max_depth or 
            n_classes == 1 or 
            n_samples < self.min_samples_split):
            # Créer une feuille
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        # Trouver la meilleure division
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_gain == 0:
            # Pas de gain, créer une feuille
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        # Diviser les données
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Vérifier les conditions sur les feuilles
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        # Construire récursivement les sous-arbres
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        if issparse(X):
            X = X.toarray()
            
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, sample, tree):
        """Prédit la classe d'un échantillon"""
        if tree['leaf']:
            return tree['value']
        
        if sample[tree['feature']] <= tree['threshold']:
            return self._predict_sample(sample, tree['left'])
        else:
            return self._predict_sample(sample, tree['right'])
    
    def predict(self, X):
        if issparse(X):
            X = X.toarray()
            
        X = check_array(X)
        return np.array([self._predict_sample(sample, self.tree) for sample in X])
    
    def get_params(self, deep=True):
        """Récupère les paramètres de l'estimateur (requis par scikit-learn)"""
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf
        }
    
    def set_params(self, **params):
        """Définit les paramètres de l'estimateur (requis par scikit-learn)"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Si le paramètre n'existe pas, lever une erreur
                raise ValueError(f"Invalid parameter {key} for estimator {self.__class__.__name__}")
        return self