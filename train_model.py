import logging
import multiprocessing
import warnings
from nltk.corpus import gutenberg, state_union, abc, conll2000, conll2002, conll2007
from string import punctuation
from nltk.corpus import brown, movie_reviews, treebank, reuters, webtext,subjectivity
from nltk.corpus import framenet as fn, genesis, rte, twitter_samples, names, inaugural
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import word2vec


abc_corp_sents = abc.sents()
frame_net_corp_sents = fn.sents()
state_union_corp_sents = state_union.sents()
subject_corp_sents = subjectivity.sents()
brown_corp_sents = brown.sents()
movie_reviews_corp_sents = movie_reviews.sents()
guttenberg_corp_sents = gutenberg.sents()
treebank_corb_sents = treebank.sents()
reuters_corp_sents = reuters.sents()
webtext_corp_sents = webtext.sents()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


discard_punctuation_and_lowercased_sents_state =  [[word.lower() for word in sent if word not in punctuation] for sent in
                                                state_union_corp_sents]





discard_punctuation_and_lowercased_sents_reviews =  [[word.lower() for word in sent if word not in punctuation] for sent in
                                                movie_reviews_corp_sents]




discard_punctuation_and_lowercased_sents_treebank =  [[word.lower() for word in sent if word not in punctuation] for sent in
                                                treebank_corb_sents]

discard_punctuation_and_lowercased_sents_gutten =  [[word.lower() for word in sent if word not in punctuation] for sent in
                                                guttenberg_corp_sents]

discard_punctuation_and_lowercased_sents_brown = [[word.lower() for word in sent if word not in punctuation] for sent in
                                                brown_corp_sents]



discard_punctuation_and_lowercased_sents_reuters = [[word.lower() for word in sent if word not in punctuation] for sent in
                                                reuters_corp_sents]


discard_punctuation_and_lowercased_sents_web_text = [[word.lower() for word in sent if word not in punctuation] for sent in
                                                 webtext_corp_sents]


discard_punctuation_and_lowercased_sents_subject = [[word.lower() for word in sent if word not in punctuation] for sent in
                                                 subject_corp_sents]



discard_punctuation_and_lowercased_sents_fn = [[word.lower() for word in sent if word not in punctuation] for sent in
                                                 frame_net_corp_sents]




compine_sents = \
    discard_punctuation_and_lowercased_sents_brown + \
    discard_punctuation_and_lowercased_sents_reviews + \
    discard_punctuation_and_lowercased_sents_treebank + \
    discard_punctuation_and_lowercased_sents_reuters + \
    discard_punctuation_and_lowercased_sents_web_text + \
    discard_punctuation_and_lowercased_sents_gutten + \
    discard_punctuation_and_lowercased_sents_subject + \
    discard_punctuation_and_lowercased_sents_state + \
    discard_punctuation_and_lowercased_sents_fn


num_features = 300
# Minimum word count threshold.
min_word_count = 1

# Number of threads to run in parallel.
#more workers, faster we train
num_workers =  multiprocessing.cpu_count()

# Context window length.
context_size = 7

# Downsample setting for frequent words.
#0 - 1e-5 is good for this
downsampling = 1e-3

# Seed for the RNG, to make the results reproducible.
#random number generator
#deterministic, good for debugging




print(len(compine_sents))

bible_kjv_word2vec_model = word2vec.Word2Vec(compine_sents, min_count=5, size=200)
#print(bible_kjv_word2vec_model.most_similar(["god"]))
print(bible_kjv_word2vec_model.most_similar(["girl"], topn=20))
#print(brown.fileids())

