from abc import ABC, abstractmethod
import sklearn
import inspect

def get_classifier_class(classifier_framework: str):
    try:
        return eval(classifier_framework)
    except NameError as e:
        raise ValueError(f"Invalid classifier framework: {classifier_framework}. Please check the input.")
    

def filter_kwargs(classifier_framework, classifier_params):
    sig = inspect.signature(classifier_framework.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered_params = {k: v for k, v in classifier_params.items() if k in valid_params}

    if len(filtered_params) == 0:
        raise ValueError(f"Invalid classifier parameters {classifier_params}. Please check the input.")
    else:
        return filtered_params


class BaseClassifier(ABC):
    @abstractmethod
    def fit(self, X, y): 
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def get_model(self): 
        pass


class SKLearnClassifier(BaseClassifier):
    def __init__(self, classifier_framework, classifier_params):
        self.classifier_framework = get_classifier_class(classifier_framework)
        self.classifier_params = classifier_params
        self.filtered_params = filter_kwargs(self.classifier_framework, self.classifier_params)
        self.model = self.classifier_framework(**self.filtered_params)
        self._ignored = set(classifier_params) - set(self.filtered_params)


    def fit(self, X, y):
        self.model.fit(X, y)


    def predict(self, X):
        return self.model.predict(X)
    

    def get_model(self):
        return self.model
    

    def ignored_params(self):
        return self._ignored
    