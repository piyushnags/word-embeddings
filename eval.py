# NLP Imports
import gensim
from gensim.test.utils import datapath
from gensim.models import Word2Vec

# Project Imports
from corpora import LeeCorpus

# Miscellaneous Imports
import pprint



def evaluate_word2vec(corpus_name: str = 'lee'):
    pp = pprint.PrettyPrinter(indent=2)
    if corpus_name != 'lee':
        raise ValueError(f"Corpus {corpus_name} not supported.")

    # Load text corpus and model
    corpus = LeeCorpus()
    model = Word2Vec(sentences=corpus)

    # Google question words evaluation
    pp.pprint(model.wv.evaluate_word_analogies(datapath('questions-words.txt')))

    # Evaluation of gensim's WS-353 dataset
    pp.pprint(model.wv.evaluate_word_pairs(datapath('wordsim353.tsv')))



if __name__ == '__main__':
    evaluate_word2vec()