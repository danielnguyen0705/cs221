import pickle
import random
from collections import defaultdict, Counter

import numpy as np


class AveragedPerceptron:
    def __init__(self):
        self.weights = {}
        self.classes = set()
        self._totals = defaultdict(float)
        self._tstamps = defaultdict(int)
        self.i = 0

    def predict(self, features):
        scores = defaultdict(float)
        for feat, value in features.items():
            if value == 0:
                continue
            weights = self.weights.get(feat)
            if not weights:
                continue
            for label, weight in weights.items():
                scores[label] += value * weight
        return max(self.classes, key=lambda label: (scores[label], label)) if self.classes else "NN"

    def update(self, truth, guess, features):
        self.i += 1
        if truth == guess:
            return
        for feat in features:
            weights = self.weights.setdefault(feat, {})
            self._update_feat(feat, truth, weights.get(truth, 0.0), 1.0)
            self._update_feat(feat, guess, weights.get(guess, 0.0), -1.0)

    def _update_feat(self, feat, label, weight, value):
        param = (feat, label)
        self._totals[param] += (self.i - self._tstamps[param]) * weight
        self._tstamps[param] = self.i
        self.weights[feat][label] = weight + value

    def average_weights(self):
        for feat, weights in list(self.weights.items()):
            new_weights = {}
            for label, weight in weights.items():
                param = (feat, label)
                total = self._totals[param] + (self.i - self._tstamps[param]) * weight
                averaged = round(total / float(self.i), 3)
                if averaged:
                    new_weights[label] = averaged
            if new_weights:
                self.weights[feat] = new_weights
            else:
                del self.weights[feat]
        self._totals = defaultdict(float)
        self._tstamps = defaultdict(int)


class POSTagger:
    START = ["-START-", "-START2-"]
    END = ["-END-", "-END2-"]

    def __init__(self):
        self.model = AveragedPerceptron()
        self.tagdict = {}
        self.classes = set()
        self.DefaultPOS = "NN"

        self.hmm_tags = []
        self.hmm_start_log = None
        self.hmm_trans_log = None
        self.hmm_emit = {}
        self.hmm_suffix = {}
        self.hmm_prefix = {}
        self.hmm_shape = {}
        self.hmm_tag_counts = None
        self.hmm_zero = None
        self.hmm_vocab = set()
        self.hmm_unk_cache = {}
        self.mfc_tag = {}
        self.mfc_freq = {}
        self.mfc_conf = {}

    def fit(self, Sents, POSs):
        self._fit_perceptron(Sents, POSs)
        self._fit_hmm(Sents, POSs)

    def predict(self, Sents):
        results = []
        for words in Sents:
            if not words:
                results.append([])
                continue
            p_tags, p_margins = self._predict_perceptron_sent(words, return_margins=True)
            h_tags = self._predict_hmm_sent(words)
            sent_tags = []
            for word, p_tag, p_margin, h_tag in zip(words, p_tags, p_margins, h_tags):
                if word not in self.hmm_vocab:
                    sent_tags.append(p_tag)
                elif self.mfc_freq.get(word, 0) >= 20 and self.mfc_conf.get(word, 0.0) >= 0.99:
                    sent_tags.append(self.mfc_tag[word])
                elif p_tag != h_tag and p_margin >= 5.5:
                    sent_tags.append(p_tag)
                else:
                    sent_tags.append(h_tag)
            results.append(sent_tags)
        return results

    def _fit_perceptron(self, Sents, POSs):
        tagged_sents = [list(zip(words, tags)) for words, tags in zip(Sents, POSs)]
        self._make_tagdict(tagged_sents)
        self.model.classes = set(self.classes)
        if self.classes:
            self.DefaultPOS = Counter(tag for sent in POSs for tag in sent).most_common(1)[0][0]
        rng = random.Random(1)
        for _ in range(2):
            for sent in tagged_sents:
                words, tags = zip(*sent)
                context = self.START + [self._normalize(w) for w in words] + self.END
                prev, prev2 = self.START
                for i, word in enumerate(words):
                    guess = self.tagdict.get(word)
                    if guess is None:
                        feats = self._get_features(i, word, context, prev, prev2)
                        guess = self.model.predict(feats)
                        self.model.update(tags[i], guess, feats)
                    prev2, prev = prev, guess
            rng.shuffle(tagged_sents)
        self.model.average_weights()

    def _predict_perceptron_sent(self, words, return_margins=False):
        context = self.START + [self._normalize(w) for w in words] + self.END
        prev, prev2 = self.START
        tags = []
        margins = []
        for i, word in enumerate(words):
            tag = self.tagdict.get(word)
            if tag is None:
                feats = self._get_features(i, word, context, prev, prev2)
                tag, margin = self._predict_perceptron_with_margin(feats)
            else:
                margin = 999.0
            tags.append(tag)
            margins.append(margin)
            prev2, prev = prev, tag
        if return_margins:
            return tags, margins
        return tags

    def _predict_perceptron_with_margin(self, features):
        scores = defaultdict(float)
        for feat, value in features.items():
            if value == 0:
                continue
            weights = self.model.weights.get(feat)
            if not weights:
                continue
            for label, weight in weights.items():
                scores[label] += value * weight
        if not self.model.classes:
            return self.DefaultPOS, 0.0
        ranked = sorted(((scores[label], label) for label in self.model.classes), reverse=True)
        if len(ranked) == 1:
            return ranked[0][1], 999.0
        return ranked[0][1], ranked[0][0] - ranked[1][0]

    def _make_tagdict(self, tagged_sents):
        counts = defaultdict(lambda: defaultdict(int))
        for sent in tagged_sents:
            for word, tag in sent:
                counts[word][tag] += 1
                self.classes.add(tag)
        self.tagdict = {}
        for word, tag_freqs in counts.items():
            tag, mode = max(tag_freqs.items(), key=lambda item: item[1])
            total = sum(tag_freqs.values())
            if total >= 20 and mode / total >= 0.97:
                self.tagdict[word] = tag

    def _normalize(self, word):
        if "-" in word and word[:1] != "-":
            return "!HYPHEN"
        if word.isdigit() and len(word) == 4:
            return "!YEAR"
        if word and word[0].isdigit():
            return "!DIGITS"
        return word.lower()

    def _get_features(self, i, word, context, prev, prev2):
        i += len(self.START)
        features = defaultdict(int)

        def add(name, *args):
            features[" ".join((name,) + tuple(args))] += 1

        add("bias")
        add("i suffix", word[-3:])
        add("i pref1", word[0] if word else "")
        add("i-1 tag", prev)
        add("i-2 tag", prev2)
        add("i tag+i-2 tag", prev, prev2)
        add("i word", context[i])
        add("i-1 tag+i word", prev, context[i])
        add("i-1 word", context[i - 1])
        add("i-1 suffix", context[i - 1][-3:])
        add("i-2 word", context[i - 2])
        add("i+1 word", context[i + 1])
        add("i+1 suffix", context[i + 1][-3:])
        add("i+2 word", context[i + 2])
        return features

    def _fit_hmm(self, Sents, POSs):
        tags = sorted(set(tag for sent in POSs for tag in sent))
        self.hmm_tags = tags
        tag_to_id = {tag: i for i, tag in enumerate(tags)}
        tag_count = len(tags)
        zero = np.zeros(tag_count, dtype=float)

        word_counts = Counter(word for sent in Sents for word in sent)
        word_tag_counts = defaultdict(lambda: np.zeros(tag_count, dtype=float))
        suffix_counts = defaultdict(lambda: np.zeros(tag_count, dtype=float))
        prefix_counts = defaultdict(lambda: np.zeros(tag_count, dtype=float))
        shape_counts = defaultdict(lambda: np.zeros(tag_count, dtype=float))
        start_counts = np.zeros(tag_count, dtype=float)
        trans_counts = np.zeros((tag_count, tag_count), dtype=float)
        tag_counts = np.zeros(tag_count, dtype=float)
        mfc_counts = defaultdict(Counter)

        for words, pos_tags in zip(Sents, POSs):
            prev = None
            for word, tag in zip(words, pos_tags):
                tag_id = tag_to_id[tag]
                tag_counts[tag_id] += 1.0
                word_tag_counts[word][tag_id] += 1.0
                mfc_counts[word][tag] += 1
                if prev is None:
                    start_counts[tag_id] += 1.0
                else:
                    trans_counts[prev, tag_id] += 1.0
                if word_counts[word] <= 2:
                    lower = word.lower()
                    for length in range(1, 5):
                        suffix_counts[lower[-length:]][tag_id] += 1.0
                    for length in range(1, 4):
                        prefix_counts[lower[:length]][tag_id] += 1.0
                    if word[:1].isupper():
                        shape_counts["INITCAP"][tag_id] += 1.0
                    if any(ch.isdigit() for ch in word):
                        shape_counts["HASDIGIT"][tag_id] += 1.0
                    if "-" in word:
                        shape_counts["HASHYPHEN"][tag_id] += 1.0
                prev = tag_id

        self.hmm_vocab = set(word_tag_counts.keys())
        self.hmm_zero = zero
        self.hmm_tag_counts = tag_counts
        self.hmm_start_log = np.log((start_counts + 0.1) / (start_counts.sum() + 0.1 * tag_count))
        self.hmm_trans_log = np.log((trans_counts + 0.1) / (trans_counts.sum(axis=1, keepdims=True) + 0.1 * tag_count))
        vocab_size = max(len(word_tag_counts), 1)
        self.hmm_emit = {
            word: np.log((counts + 0.001) / (tag_counts + 0.001 * vocab_size))
            for word, counts in word_tag_counts.items()
        }
        self.hmm_suffix = dict(suffix_counts)
        self.hmm_prefix = dict(prefix_counts)
        self.hmm_shape = dict(shape_counts)
        self.hmm_unk_cache = {}

        self.mfc_tag = {}
        self.mfc_freq = {}
        self.mfc_conf = {}
        for word, counter in mfc_counts.items():
            tag, freq = counter.most_common(1)[0]
            total = sum(counter.values())
            self.mfc_tag[word] = tag
            self.mfc_freq[word] = total
            self.mfc_conf[word] = freq / total

    def _unknown_log_distribution(self, word):
        cached = self.hmm_unk_cache.get(word)
        if cached is not None:
            return cached
        zero = self.hmm_zero
        lower = word.lower()
        scores = np.ones(len(self.hmm_tags), dtype=float) + 0.001 * self.hmm_tag_counts
        for length, weight in ((4, 3.0), (3, 2.5), (2, 1.5), (1, 0.5)):
            scores += weight * self.hmm_suffix.get(lower[-length:], zero)
        for length, weight in ((3, 0.8), (2, 0.4), (1, 0.2)):
            scores += weight * self.hmm_prefix.get(lower[:length], zero)
        if word[:1].isupper():
            scores += 1.2 * self.hmm_shape.get("INITCAP", zero)
        if any(ch.isdigit() for ch in word):
            scores += 2.0 * self.hmm_shape.get("HASDIGIT", zero)
        if "-" in word:
            scores += 1.0 * self.hmm_shape.get("HASHYPHEN", zero)
        scores = scores / scores.sum()
        result = np.log(scores)
        self.hmm_unk_cache[word] = result
        return result

    def _predict_hmm_sent(self, words):
        if not words:
            return []
        tag_count = len(self.hmm_tags)
        first_emit = self.hmm_emit.get(words[0])
        if first_emit is None:
            first_emit = self._unknown_log_distribution(words[0])
        scores = self.hmm_start_log + first_emit
        backpointers = []
        tag_indices = np.arange(tag_count)

        for word in words[1:]:
            emission = self.hmm_emit.get(word)
            if emission is None:
                emission = self._unknown_log_distribution(word)
            values = scores[:, None] + self.hmm_trans_log
            best_prev = values.argmax(axis=0)
            scores = values[best_prev, tag_indices] + emission
            backpointers.append(best_prev)

        last = int(scores.argmax())
        ids = [last]
        for bp in reversed(backpointers):
            last = int(bp[last])
            ids.append(last)
        ids.reverse()
        return [self.hmm_tags[i] for i in ids]

    def save(self):
        with open("model.mdl", "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self=None):
        with open("model.mdl", "rb") as f:
            return pickle.load(f)
