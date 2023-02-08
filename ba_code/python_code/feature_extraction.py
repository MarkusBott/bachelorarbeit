from time import perf_counter
from numpy_da import DynamicArray

import gensim
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


#tf-idf vectorization. The values for min_ and max_df were created empirically
def tfidf_vectorize(data):
    start = perf_counter()
    vectorizer = TfidfVectorizer()
    vectorizer.min_df = 2 #the minimum document frequency of a term 
    #-> a word is only vectorized, if it contains in more than one document
    vectorizer.max_df = 0.7 #the maximum document frequency of a term
    #-> a word is only vectorized, if it contains in equal or
    #less 70% of the documents
    X = vectorizer.fit_transform(data)
    end = perf_counter()
    print(f'It took {end-start} second(s) to finish TF-IDF feature extraction.')
    return X

#creates an iterable object suitable for doc2vec
def __tagged_doc(data):
    for i, list_of_words in enumerate(data):
      yield gensim.models.doc2vec.TaggedDocument(list_of_words, [i])

#doc2vec vectorization
def doc2vec_vectorize(data, v_size):
    start = perf_counter()
    tokenized = data.apply(lambda text : text.split())
    data_for_training = list(__tagged_doc(tokenized))
    #vector_size describes the dimension of the output vector, 
    #min_count means the same as min_df in tfidf_vectorize
    #epochs is the number of epochs the model is training
    model = gensim.models.doc2vec.Doc2Vec(vector_size=v_size,min_count=2, epochs=30)
    model.build_vocab(data_for_training) #build the vocabulary
    end = perf_counter()
    #String contains: It took {end-start} second(s) 
    #to create the model and build the vocabulary.
    print(f'It took {end-start} second(s) to create the model and build the vocabulary.')
    #train the model -> every document gets a 100d vector
    model.train(data_for_training, 
            total_examples=model.corpus_count, 
            epochs=model.epochs)
    #put the vectors in a numpy array for output
    text_list = np.empty((len(tokenized),v_size))
    c = 0
    for mail in tokenized:
        text_list[c] = model.infer_vector(mail)
        c += 1
    end = perf_counter()
    print(f'It took {end-start} second(s) to finish doc2vec.')
    return text_list