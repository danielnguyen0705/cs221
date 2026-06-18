import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import VotingClassifier

class Model:
    def __init__(self):
        word_vect = TfidfVectorizer(
            analyzer='word',
            lowercase=True,
            ngram_range=(1, 2),
            max_df=0.85,
            min_df=5,
            sublinear_tf=True,
            binary=True,
            max_features=50000,
        )

        char_vect = TfidfVectorizer(
            analyzer='char_wb',
            lowercase=True,
            ngram_range=(3, 5),
            max_df=0.85,
            min_df=10,
            sublinear_tf=True,
            binary=True,
            max_features=30000,
        )

        self.pipeline = Pipeline([
            ('features', FeatureUnion([
                ('word', word_vect),
                ('char', char_vect),
            ])),
            ('clf', VotingClassifier(
                estimators=[
                    ('lr', LogisticRegression(
                        C=4.0,
                        class_weight='balanced',
                        solver='lbfgs',
                        max_iter=1500,
                        random_state=42,
                        n_jobs=1,
                    )),
                    ('svc', LinearSVC(
                        C=0.5,
                        class_weight='balanced',
                        max_iter=2000,
                        random_state=42,
                    )),
                    ('cnb', ComplementNB(alpha=0.3)),
                ],
                voting='hard',
                weights=[3, 3, 1],
            )),
        ])

    def _clean_text(self, text):
        if text is None or (isinstance(text, float) and np.isnan(text)):
            return ""
        return str(text).strip().lower()

    def fit(self, Train_Text, Train_Label):
        processed_train = [self._clean_text(t) for t in Train_Text]
        self.pipeline.fit(processed_train, Train_Label)

    def predict(self, Test_Text):
        processed_test = [self._clean_text(t) for t in Test_Text]
        predictions = self.pipeline.predict(processed_test)
        return predictions.tolist()