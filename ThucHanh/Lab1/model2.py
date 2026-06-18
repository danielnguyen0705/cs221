import re
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV

# ── VADER (lexicon-based sentiment scores) ────────────────────────────────────
try:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    _HAS_VADER = True
except Exception:
    _HAS_VADER = False

# ── Negation vocabulary ───────────────────────────────────────────────────────
_NEG_WORDS = frozenset({
    'not', 'no', 'never', 'nor', 'nothing', 'nobody', 'nowhere',
    'neither', 'cannot', 'hardly', 'barely', 'scarcely', 'seldom',
    'without', 'lack', 'lacks', 'lacking', 'failed', 'fails', 'fail',
})
_CLAUSE_BOUNDS = frozenset({'but', 'however', 'although', 'though', 'yet',
                            'nevertheless', 'nonetheless', 'despite'})

# Pre-compiled contraction patterns (order matters — longer first)
_CONTRACTIONS = [
    (re.compile(r"\bwon't\b"),      "will not"),
    (re.compile(r"\bcan't\b"),      "cannot"),
    (re.compile(r"\bshan't\b"),     "shall not"),
    (re.compile(r"n't\b"),          " not"),
    (re.compile(r"'re\b"),          " are"),
    (re.compile(r"'ve\b"),          " have"),
    (re.compile(r"'ll\b"),          " will"),
    (re.compile(r"'d\b"),           " would"),
    (re.compile(r"'m\b"),           " am"),
]


def _preprocess(text: str) -> str:
    """
    Expand contractions + mark tokens after negation words with NOT_ prefix.
    Negation scope ends after 4 tokens or at clause-boundary words.
    """
    if not text:
        return ""
    text = str(text).lower()
    for pat, repl in _CONTRACTIONS:
        text = pat.sub(repl, text)

    tokens = text.split()
    out = []
    neg = False
    scope = 0
    for tok in tokens:
        word = tok.strip('.,!?;:')
        if word in _NEG_WORDS:
            neg, scope = True, 0
            out.append(tok)
        elif tok in {'.', '!', '?', ';'} or word in _CLAUSE_BOUNDS:
            neg = False
            out.append(tok)
        elif neg:
            out.append('NOT_' + tok)
            scope += 1
            if scope >= 4:
                neg = False
        else:
            out.append(tok)
    return ' '.join(out)


class Model:
    """
    3-class English sentiment classifier.

    Architecture
    ────────────
    Features  : word TF-IDF (1–2-gram) + char_wb TF-IDF (3–5-gram)
                + 4 VADER lexicon scores  (if nltk available)
    Ensemble  : soft-vote of three classifiers
                  • CalibratedLinearSVC  (weight 3) – strong linear separator
                  • LogisticRegression   (weight 2) – calibrated probabilities
                  • ComplementNB         (weight 1) – robust on sparse text
    Key ideas : negation marking in preprocessing, VADER as external lexicon,
                soft voting exploits calibrated confidence scores.
    """

    def __init__(self):
        # ── Vectorisers ───────────────────────────────────────────────────────
        self._wvec = TfidfVectorizer(
            analyzer='word', lowercase=True,
            ngram_range=(1, 2), max_df=0.90, min_df=2,
            sublinear_tf=True, max_features=120_000,
        )
        self._cvec = TfidfVectorizer(
            analyzer='char_wb', lowercase=True,
            ngram_range=(3, 5), max_df=0.90, min_df=3,
            sublinear_tf=True, max_features=80_000,
        )

        # ── Classifiers ───────────────────────────────────────────────────────
        # CalibratedClassifierCV: trains 3-fold CV internally so it can output
        # proper probabilities from LinearSVC; final model uses all training data.
        self._svc = CalibratedClassifierCV(
            LinearSVC(
                C=0.5, class_weight='balanced',
                max_iter=3000, random_state=42,
            ),
            cv=3, method='sigmoid',
        )
        self._lr = LogisticRegression(
            C=2.0, class_weight='balanced',
            max_iter=2000, solver='lbfgs',
            n_jobs=-1, random_state=42,
        )
        self._cnb = ComplementNB(alpha=0.3)

        # ── VADER ─────────────────────────────────────────────────────────────
        self._sia = None
        if _HAS_VADER:
            try:
                self._sia = SentimentIntensityAnalyzer()
            except Exception:
                pass

        self._classes = None     # filled during fit()
        self._has_vader = False  # flag set during fit()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _vader_matrix(self, texts):
        """Returns (n, 4) sparse matrix of VADER neg/neu/pos/compound scores."""
        if self._sia is None:
            return None
        rows = []
        for t in texts:
            s = self._sia.polarity_scores(str(t) if t else "")
            rows.append([s['neg'], s['neu'], s['pos'], s['compound']])
        return sp.csr_matrix(np.array(rows, dtype=np.float32))

    def _build_features(self, raw_texts, processed_texts, fit: bool):
        """
        Combine word TF-IDF + char TF-IDF → `tfidf_feat`  (always non-negative)
        Then optionally append VADER scores  → `full_feat`  (for SVC + LR)
        CNB is trained on `tfidf_feat` only (requires non-negative input).
        """
        if fit:
            wf = self._wvec.fit_transform(processed_texts)
            cf = self._cvec.fit_transform(processed_texts)
        else:
            wf = self._wvec.transform(processed_texts)
            cf = self._cvec.transform(processed_texts)

        tfidf_feat = sp.hstack([wf, cf], format='csr')

        vf = self._vader_matrix(raw_texts)
        if vf is not None:
            full_feat = sp.hstack([tfidf_feat, vf], format='csr')
        else:
            full_feat = tfidf_feat

        return tfidf_feat, full_feat

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, Train_Text, Train_Label):
        processed = [_preprocess(t) for t in Train_Text]
        tfidf_feat, full_feat = self._build_features(
            Train_Text, processed, fit=True
        )
        self._has_vader = full_feat.shape[1] != tfidf_feat.shape[1]

        # Each classifier trains on what suits it best
        self._svc.fit(full_feat, Train_Label)   # SVC handles negative values fine
        self._lr.fit(full_feat, Train_Label)    # LR handles negative values fine
        self._cnb.fit(tfidf_feat, Train_Label)  # CNB requires non-negative → TF-IDF only

        self._classes = self._lr.classes_

    def predict(self, Test_Text):
        processed = [_preprocess(t) for t in Test_Text]
        tfidf_feat, full_feat = self._build_features(
            Test_Text, processed, fit=False
        )

        p_svc = self._svc.predict_proba(full_feat)
        p_lr  = self._lr.predict_proba(full_feat)
        p_cnb = self._cnb.predict_proba(tfidf_feat)

        # Weighted soft vote (weights tuned empirically: SVC > LR > CNB)
        combined = 3.0 * p_svc + 2.0 * p_lr + 1.0 * p_cnb
        idx = np.argmax(combined, axis=1)
        return [int(self._classes[i]) for i in idx]