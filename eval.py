# NLP Imports
import gensim
from gensim.test.utils import datapath
from gensim.models import Word2Vec

# Project Imports
from corpora import LeeCorpus

# Miscellaneous Imports
import pprint



def evaluate_word2vec(corpus_name: str = 'lee'):
    '''
    Description:
        Function to evaluate pretrained word2vec model
        on a text corpus. Defaults to lee corpus
    
    Args:
        corpus_name: name of corpus to be used for evaluation
    
    Returns:
        None
    '''
    if corpus_name != 'lee':
        raise ValueError(f"Corpus {corpus_name} not supported.")

    # Load text corpus and model
    corpus = LeeCorpus()
    model = Word2Vec(sentences=corpus)

    pp = pprint.PrettyPrinter(indent=2)
    # Google question words evaluation
    pp.pprint(model.wv.evaluate_word_analogies(datapath('questions-words.txt')))

    # Evaluation of gensim's WS-353 dataset
    pp.pprint(model.wv.evaluate_word_pairs(datapath('wordsim353.tsv')))



if __name__ == '__main__':
    # TODO: Update with cfg or arg parse at some point
    evaluate_word2vec()