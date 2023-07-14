# NLP Imports
import gensim
from gensim.test.utils import datapath



class LeeCorpus():
    '''
    Wrapper class for the Lee Corpus
    that streams lines instead of loading
    full corpus in memory
    '''
    def __iter__(self):
        path = datapath('lee_background.cor')
        # Use gensim's open
        for line in gensim.utils.open(path):
            yield gensim.utils.simple_preprocess(line)