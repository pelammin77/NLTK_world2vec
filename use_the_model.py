import os
import gensim.models.word2vec as w2v


model = w2v.Word2Vec.load(os.path.join("trained_model", "corpus2vec.w2v"))

print(model.most_similar(['horse'], topn=15))
print(model.most_similar(positive=['emperor', 'girl'], negative=['man'], topn=10))