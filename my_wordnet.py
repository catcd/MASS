from nltk.corpus import wordnet as wn
from nltk import WordNetLemmatizer


class WordNet:
    lemmer = None

    @staticmethod
    def get_wordnet_pos(treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
        """
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return wn.NOUN

    @staticmethod
    def lemmatize(word, pos):
        if WordNet.lemmer is None:
            WordNet.lemmer = WordNetLemmatizer()
        return WordNet.lemmer.lemmatize(word, WordNet.get_wordnet_pos(pos))
