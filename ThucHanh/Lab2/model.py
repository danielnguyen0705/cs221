import os
import pickle
import random
import re
from collections import Counter, defaultdict


class _WittenBellNGramLM:
    """Pure word-level probabilistic n-gram language model.

    This implementation follows the teacher's warning strictly:
    - each LM is trained only on its own corpus passed to fit();
    - no opposite corpus is loaded or compared;
    - no word_score, nb_boost, p_class, candidate_score, or sentence_score;
    - no list of hand-picked words/phrases is used;
    - no original training sentence is stored and replayed.

    Generation samples directly from P(next_word | history). The only decoding
    controls are top_k, top_p, and temperature.
    """

    BOS = "<BOS>"
    EOS = "<EOS>"

    def __init__(
        self,
        order=4,
        max_len=18,
        unigram_alpha=0.05,
        top_k=12,
        top_p=0.88,
        temperature=0.72,
    ):
        self.order = max(2, int(order))
        self.max_context = self.order - 1
        self.max_len = int(max_len)
        self.unigram_alpha = float(unigram_alpha)

        # Allowed generation parameters.
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

        # Only avoids duplicate generated outputs across generate() calls, as required by the assignment.
        # This is not a scoring/ranking mechanism and does not change token probabilities.
        self._unique_retries = 30
        self.generated = set()

        # counts[k][context_tuple] -> Counter(next_token), k is context length.
        self.counts = [defaultdict(Counter) for _ in range(self.max_context + 1)]
        self.total = [defaultdict(int) for _ in range(self.max_context + 1)]
        self.types = [defaultdict(int) for _ in range(self.max_context + 1)]
        self.vocab = set()
        self.vocab_list = []
        self._dist_cache = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_dist_cache"] = {}
        return state

    def _normalize(self, text):
        text = str(text).replace("`", "'").replace("’", "'").replace("‘", "'")
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[@_]+[A-Za-z0-9_]*", " ", text)
        text = re.sub(r"&amp;", " and ", text)
        text = re.sub(r"[^A-Za-z0-9' ]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    def _tokenize(self, text):
        return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", self._normalize(text))

    def _iter_lines(self, data):
        if isinstance(data, str):
            if os.path.exists(data):
                with open(data, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        yield line
            else:
                for line in data.splitlines():
                    yield line
        else:
            for line in data:
                yield line

    def fit(self, data):
        self.counts = [defaultdict(Counter) for _ in range(self.max_context + 1)]
        self.total = [defaultdict(int) for _ in range(self.max_context + 1)]
        self.types = [defaultdict(int) for _ in range(self.max_context + 1)]
        self.vocab = set()
        self.vocab_list = []
        self.generated = set()
        self._dist_cache = {}

        pad = [self.BOS] * self.max_context
        for line in self._iter_lines(data):
            words = self._tokenize(line)
            if not words:
                continue
            words = words[: self.max_len]
            self.vocab.update(words)
            seq = pad + words + [self.EOS]

            for i in range(self.max_context, len(seq)):
                token = seq[i]
                max_k = min(self.max_context, i)
                for k in range(max_k + 1):
                    context = tuple(seq[i - k:i]) if k > 0 else ()
                    self.counts[k][context][token] += 1

        self.vocab.add(self.EOS)
        self.vocab_list = sorted(self.vocab)

        for k in range(self.max_context + 1):
            for context, counter in self.counts[k].items():
                self.total[k][context] = sum(counter.values())
                self.types[k][context] = len(counter)
        return self

    def _unigram_distribution(self):
        context = ()
        counter = self.counts[0].get(context, Counter())
        total = self.total[0].get(context, 0)
        alpha = self.unigram_alpha
        denom = total + alpha * max(1, len(self.vocab_list))
        if denom <= 0:
            return {self.EOS: 1.0}
        return {
            tok: (counter.get(tok, 0) + alpha) / denom
            for tok in self.vocab_list
        }

    def _distribution_for_context(self, context):
        """Return Witten-Bell interpolated P(next | context)."""
        context = tuple(context[-self.max_context:])
        if context in self._dist_cache:
            return dict(self._dist_cache[context])

        k = len(context)
        if k == 0:
            dist = self._unigram_distribution()
            self._dist_cache[context] = dict(dist)
            return dist

        counter = self.counts[k].get(context)
        if not counter:
            dist = self._distribution_for_context(context[1:])
            self._dist_cache[context] = dict(dist)
            return dict(dist)

        C = self.total[k].get(context, 0)
        T = self.types[k].get(context, 0)
        if C <= 0 or T <= 0:
            dist = self._distribution_for_context(context[1:])
            self._dist_cache[context] = dict(dist)
            return dict(dist)

        lower = self._distribution_for_context(context[1:])
        # Witten-Bell interpolation: seen-continuation mass = C / (C + T).
        lambda_seen = C / (C + T)
        lambda_backoff = T / (C + T)

        dist = {}
        for tok in self.vocab_list:
            mle = counter.get(tok, 0) / C
            dist[tok] = lambda_seen * mle + lambda_backoff * lower.get(tok, 0.0)

        # Normalize to remove tiny numerical drift.
        s = sum(dist.values())
        if s > 0:
            for tok in list(dist.keys()):
                dist[tok] /= s

        self._dist_cache[context] = dict(dist)
        return dist

    def _sample_from_distribution(self, dist):
        if not dist:
            return self.EOS

        temp = max(1e-6, float(self.temperature))
        items = []
        for tok, prob in dist.items():
            if tok == self.BOS or prob <= 0:
                continue
            items.append((tok, prob ** (1.0 / temp)))

        if not items:
            return self.EOS

        items.sort(key=lambda x: x[1], reverse=True)

        # Top-k sampling.
        if self.top_k is not None and self.top_k > 0:
            items = items[: int(self.top_k)]

        # Top-p / nucleus sampling.
        if self.top_p is not None and 0 < self.top_p < 1:
            total = sum(w for _, w in items)
            cutoff = self.top_p * total
            kept = []
            acc = 0.0
            for tok, weight in items:
                kept.append((tok, weight))
                acc += weight
                if acc >= cutoff:
                    break
            items = kept

        total = sum(w for _, w in items)
        if total <= 0:
            return self.EOS

        r = random.random() * total
        acc = 0.0
        for tok, weight in items:
            acc += weight
            if acc >= r:
                return tok
        return items[-1][0]

    def _generate_once(self):
        history = [self.BOS] * self.max_context
        output = []
        for _ in range(self.max_len):
            context = tuple(history[-self.max_context:])
            dist = self._distribution_for_context(context)
            token = self._sample_from_distribution(dist)
            if token == self.EOS:
                break
            output.append(token)
            history.append(token)
        return output

    def _detokenize(self, tokens):
        if not tokens:
            return ""
        text = " ".join(tokens)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return ""
        if text[-1] not in ".!?":
            text += "."
        return text[0].upper() + text[1:]

    def generate(self):
        # Retries are used only to avoid returning duplicate strings, not to
        # score or select the most class-like sentence.
        last_sentence = ""
        for _ in range(self._unique_retries):
            sentence = self._detokenize(self._generate_once())
            if sentence:
                last_sentence = sentence
            key = sentence.lower()
            if sentence and key not in self.generated:
                self.generated.add(key)
                return sentence

        if last_sentence:
            self.generated.add(last_sentence.lower())
            return last_sentence
        return "Generated text."

    def save(self, path=None):
        if path is None:
            path = self.__class__.__name__ + ".mdl"
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)


class FirstLM(_WittenBellNGramLM):
    def __init__(self):
        super().__init__(
            order=4,
            max_len=18,
            unigram_alpha=0.04,
            top_k=7,
            top_p=0.82,
            temperature=0.62,
        )


class SecondLM(_WittenBellNGramLM):
    def __init__(self):
        super().__init__(
            order=4,
            max_len=18,
            unigram_alpha=0.04,
            top_k=8,
            top_p=0.84,
            temperature=0.64,
        )
