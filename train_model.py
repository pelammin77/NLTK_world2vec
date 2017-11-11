import logging
import os
import codecs
import glob
import re
import multiprocessing
import warnings
from nltk.corpus import gutenberg, state_union, abc, conll2000, conll2002, conll2007
#from nltk.tokenize import sent_tokenize
import nltk
from string import punctuation
from nltk.corpus import brown, movie_reviews, treebank, reuters, webtext,subjectivity
from nltk.corpus import framenet as fn, genesis, rte, twitter_samples, names, inaugural
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import word2vec

def sentence_to_wordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words




book_filenames = sorted(glob.glob("books/*.*"))
print("books file names", book_filenames)
books_raw = u""
for book_filename in book_filenames:
    print("Reading '{0}'...".format(book_filename))
    with codecs.open(book_filename, "r", "utf-8") as book_file:
        books_raw += book_file.read()
    print("Corpus is now {0} characters long".format(len(books_raw)))
    print()
    books_raw = books_raw.lower()




tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
raw_sentences = tokenizer.tokenize(books_raw)

book_sentences = []
for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
        book_sentences.append(sentence_to_wordlist(raw_sentence))

#print(raw_sentences[5])
#print(book_sentences[5])


conll2000_corp_sents = conll2000.sents()
print("condll2000 to sents")
conll2002_corp_sents = conll2002.sents()
print("conll2002 to sents")

conll2007_corp_sents = conll2007.sents()
print("condll2007 to sents")
inaugural_corp_sents = inaugural.sents()
print("inaugural to sents")
abc_corp_sents = abc.sents()
print("ABC to sentences")
genesis_corp_sents = genesis.sents()
print("Genesis to sents")
frame_net_corp_sents = fn.sents()
print("Frame_net to sents")
state_union_corp_sents = state_union.sents()
print('state union to sents')
subject_corp_sents = subjectivity.sents()
print('Subjectvity to sents')
brown_corp_sents = brown.sents()
print("Brown corpus to sents")
movie_reviews_corp_sents = movie_reviews.sents()
print("Movie reviews to sents ")
guttenberg_corp_sents = gutenberg.sents()
print("Guttenberg to sents")
treebank_corb_sents = treebank.sents()
print("Freebank to sents")
reuters_corp_sents = reuters.sents()
print("Reuters to sents")
webtext_corp_sents = webtext.sents()
print("Webtext to sents")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("Cleaning data ...")


discard_punctuation_and_lowercased_sents_condll2007 =  [[word.lower() for word in sent if word not in punctuation] for sent in
                                                        conll2007_corp_sents]



discard_punctuation_and_lowercased_sents_condll2000 =  [[word.lower() for word in sent if word not in punctuation] for sent in
                                                        conll2000_corp_sents]


discard_punctuation_and_lowercased_sents_condll2002 =  [[word.lower() for word in sent if word not in punctuation] for sent in
                                                        conll2002_corp_sents]





discard_punctuation_and_lowercased_sents_state =  [[word.lower() for word in sent if word not in punctuation] for sent in
                                                state_union_corp_sents]

discard_punctuation_and_lowercased_sents_abc =  [[word.lower() for word in sent if word not in punctuation] for sent in
                                                abc_corp_sents]




discard_punctuation_and_lowercased_sents_reviews =  [[word.lower() for word in sent if word not in punctuation] for sent in
                                                movie_reviews_corp_sents]


discard_punctuation_and_lowercased_sents_genesis =  [[word.lower() for word in sent if word not in punctuation] for sent in
                                                genesis_corp_sents]



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


discard_punctuation_and_lowercased_sents_inaugural =  [[word.lower() for word in sent if word not in punctuation] for sent in
                                                inaugural_corp_sents]




compine_sents = book_sentences +\
    discard_punctuation_and_lowercased_sents_brown + \
    discard_punctuation_and_lowercased_sents_reviews + \
    discard_punctuation_and_lowercased_sents_treebank + \
    discard_punctuation_and_lowercased_sents_reuters + \
    discard_punctuation_and_lowercased_sents_web_text + \
    discard_punctuation_and_lowercased_sents_gutten + \
    discard_punctuation_and_lowercased_sents_subject + \
    discard_punctuation_and_lowercased_sents_state + \
    discard_punctuation_and_lowercased_sents_fn + \
    discard_punctuation_and_lowercased_sents_abc + \
    discard_punctuation_and_lowercased_sents_genesis + \
    discard_punctuation_and_lowercased_sents_inaugural + \
    discard_punctuation_and_lowercased_sents_condll2000 + \
    discard_punctuation_and_lowercased_sents_condll2002 + \
    discard_punctuation_and_lowercased_sents_condll2007



#
# num_features = 300
# # Minimum word count threshold.
# min_word_count = 1
#
# # Number of threads to run in parallel.
# #more workers, faster we train
# num_workers =  multiprocessing.cpu_count()
#
# # Context window length.
# context_size = 7
#
# # Downsample setting for frequent words.
# #0 - 1e-5 is good for this
# downsampling = 1e-3
#
# # Seed for the RNG, to make the results reproducible.
# #random number generator
# #deterministic, good for debugging
# seed = 1


nltk_corpus_word2vec_model = word2vec.Word2Vec(compine_sents, min_count=5, size=200, workers=4)

#print(compine_sents)

if not os.path.exists("trained_model"):
    os.makedirs("trained_model")


nltk_corpus_word2vec_model.save(os.path.join("trained_model", "corpus2vec.w2v"))




print(nltk_corpus_word2vec_model.most_similar(["lord"], topn=20))
#print(brown.fileids())

