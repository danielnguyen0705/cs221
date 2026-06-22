import pickle
from collections import defaultdict, Counter

import numpy as np
import scipy.sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron


class POSTagger:
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.tagdict = {}
        self.word_majority = {}
        self.lower_majority = {}
        self.default_tag = "NN"

    def _normalize(self, token):
        if not token:
            return token
        if "-" in token and token[:1] != "-":
            return "<HYPHEN>"
        if token.isdigit() and len(token) == 4:
            return "<YEAR>"
        if token and token[0].isdigit():
            return "<NUMBER>"
        return token.lower()

    def _shape(self, token):
        result = []
        for ch in token:
            if ch.isupper():
                code = "X"
            elif ch.islower():
                code = "x"
            elif ch.isdigit():
                code = "d"
            elif ch in "-_/":
                code = "-"
            elif ch in ".,;:!?":
                code = "."
            else:
                code = ch
            if not result or result[-1] != code:
                result.append(code)
        return "".join(result[:8]) if result else "EMPTY"

    def _build_lexicon(self, Sents, POSs):
        word_counts = defaultdict(Counter)
        lower_counts = defaultdict(Counter)
        tag_counts = Counter()

        for sent, tags in zip(Sents, POSs):
            for word, tag in zip(sent, tags):
                word_counts[word][tag] += 1
                lower_counts[word.lower()][tag] += 1
                tag_counts[tag] += 1

        if tag_counts:
            self.default_tag = tag_counts.most_common(1)[0][0]

        self.word_majority = {
            word: counts.most_common(1)[0][0]
            for word, counts in word_counts.items()
        }

        self.lower_majority = {
            word: counts.most_common(1)[0][0]
            for word, counts in lower_counts.items()
        }

        self.tagdict = {}
        for word, counts in word_counts.items():
            tag, freq = counts.most_common(1)[0]
            total = sum(counts.values())

            if total >= 20 and freq / total >= 0.985:
                self.tagdict[word] = tag
            elif total >= 5 and freq / total == 1.0:
                self.tagdict[word] = tag

    def _token_features(self, sent, idx):
        token = sent[idx]
        lower = token.lower()
        norm = self._normalize(token)

        prev1 = sent[idx - 1] if idx > 0 else "<BOS>"
        prev2 = sent[idx - 2] if idx > 1 else "<BOS2>"
        next1 = sent[idx + 1] if idx + 1 < len(sent) else "<EOS>"
        next2 = sent[idx + 2] if idx + 2 < len(sent) else "<EOS2>"

        p1_norm = self._normalize(prev1)
        p2_norm = self._normalize(prev2)
        n1_norm = self._normalize(next1)
        n2_norm = self._normalize(next2)

        feats = {
            "bias": 1,

            "tok.norm": norm,
            "tok.lower": lower,
            "tok.shape": self._shape(token),

            "tok.suf1": lower[-1:],
            "tok.suf2": lower[-2:],
            "tok.suf3": lower[-3:],
            "tok.suf4": lower[-4:],

            "tok.pre1": lower[:1],
            "tok.pre2": lower[:2],
            "tok.pre3": lower[:3],

            "tok.title": token.istitle(),
            "tok.upper": token.isupper(),
            "tok.digit": token.isdigit(),
            "tok.alpha": token.isalpha(),
            "tok.hyphen": "-" in token,
            "tok.has_digit": any(ch.isdigit() for ch in token),
            "tok.has_upper": any(ch.isupper() for ch in token),

            "p1.word": p1_norm,
            "p2.word": p2_norm,
            "n1.word": n1_norm,
            "n2.word": n2_norm,

            "p1.shape": self._shape(prev1),
            "n1.shape": self._shape(next1),

            "p1+tok": p1_norm + "+" + norm,
            "tok+n1": norm + "+" + n1_norm,

            "p1.suf3": prev1.lower()[-3:],
            "n1.suf3": next1.lower()[-3:],
        }

        if idx == 0:
            feats["BOS"] = True
        if idx == len(sent) - 1:
            feats["EOS"] = True

        if token in self.word_majority:
            feats["word_majority"] = self.word_majority[token]

        if lower in self.lower_majority:
            feats["lower_majority"] = self.lower_majority[lower]

        return feats

    def _sentence_features(self, sent):
        return [self._token_features(sent, i) for i in range(len(sent))]

    def _fix_sparse_index(self, matrix):
        if scipy.sparse.issparse(matrix):
            matrix = matrix.tocsc()
            matrix.indices = matrix.indices.astype(np.int32)
            matrix.indptr = matrix.indptr.astype(np.int32)

            matrix = matrix.tocsr()
            matrix.indices = matrix.indices.astype(np.int32)
            matrix.indptr = matrix.indptr.astype(np.int32)

        return matrix

    def fit(self, Sents, POSs):
        self._build_lexicon(Sents, POSs)

        X_dicts = []
        y = []

        for sent, tags in zip(Sents, POSs):
            for i, tag in enumerate(tags):
                X_dicts.append(self._token_features(sent, i))
                y.append(tag)

        self.vectorizer = DictVectorizer(sparse=True)
        X = self.vectorizer.fit_transform(X_dicts)
        X = self._fix_sparse_index(X)

        self.classifier = Perceptron(
            max_iter=50,
            tol=1e-4,
            eta0=1.0,
            random_state=42
        )

        self.classifier.fit(X, y)

    def predict(self, Sents):
        results = []

        for sent in Sents:
            if not sent:
                results.append([])
                continue

            X_dicts = self._sentence_features(sent)
            X = self.vectorizer.transform(X_dicts)
            X = self._fix_sparse_index(X)

            pred = list(self.classifier.predict(X))

            for i, word in enumerate(sent):
                if word in self.tagdict:
                    pred[i] = self.tagdict[word]

            results.append(pred)

        return results

    def load(self=None):
        with open("model.mdl", "rb") as f:
            return pickle.load(f)

    def save(self):
        with open("model.mdl", "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)