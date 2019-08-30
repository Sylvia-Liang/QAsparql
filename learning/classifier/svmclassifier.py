from learning.classifier.classifier import Classifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif, mutual_info_classif, chi2

wordnet_lemmatizer = WordNetLemmatizer()


class Lemmarize(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new = []
        for sentence in X:
            token_words = word_tokenize(sentence)
            stem_sentence = []
            for word in token_words:
                stem_sentence.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
            new.append(" ".join(stem_sentence))
        return new


class POS(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        spacy.prefer_gpu()
        nlp = spacy.load("en_core_web_lg")
        new = []
        for sentence in X:
            doc = nlp(sentence)
            json_doc = doc.to_json()
            token = json_doc['tokens']
            tag = []
            for t in token:
                tag.append(t['tag'])
            new.append(" ".join(tag))
        return new


class SVMClassifier(Classifier):
    def __init__(self, model_file_path=None):
        super(SVMClassifier, self).__init__(model_file_path)
        self.pipeline = Pipeline([
                                  ('lemma', Lemmarize()),
                                  ('tf-idf', TfidfVectorizer(max_df=0.9, min_df=3, max_features=2000, ngram_range=(1,4))),
                                  ('svm', RandomForestClassifier(n_estimators=150, max_depth=150, criterion='gini', random_state=42))])
        self.parameters = {
                           'svm__max_features': ('sqrt', 'log2')
        }

