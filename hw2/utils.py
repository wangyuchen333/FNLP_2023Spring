import re
import csv
import random
import math
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

csv.field_size_limit(500 * 1024 * 1024)

def tokenize(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)

    # Remove stop words
    stopwords = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stopwords]

    # Stem the words
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


def preprocess(text):
    # Clean the text
    text = clean_str(text)

    # Tokenize the text
    tokens = tokenize(text)

    # Join the tokens back into a string
    text = ' '.join(tokens)

    return text



def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

def read_csv(filename):
    '''
    Get text features and labels from a csv file
    '''
    with open(filename,'r',) as f:
        reader = csv.reader(f)
        items = []
        labels = []
        for row in reader:
            if row[0] == 'data':
                continue
            item,label = row[0],row[1]
            item = preprocess(item)
            label = int(label)
            items.append(item)
            labels.append(label)
            
    return items,labels



def build_vocab(items):
    '''
    Build the vocabulary from a training csv file
    '''
    # stopwords = get_stopwords()
    dictionary = {}
    for item in items:
        words = item.split()
        for word in words:
            dictionary[word] = dictionary.get(word,0) + 1 
    dictionary = sorted(dictionary.items(),key=lambda item:item[1],reverse=True)
    return dictionary


def compute_tfidf(corpus, vocab):
    # Compute the document frequency of each word in the corpus
    df = defaultdict(int, {word:0 for word in vocab})
    for id, doc in enumerate(corpus):
        words=doc.split()
        for word in set(words):
            df[word] += 1

    # Compute the inverse document frequency of each word in the vocab
    idf = {}
    N = len(corpus)
    for word in vocab:
        if df[word]!=0:
            idf[word] = math.log(N / df[word])

    # Compute the TF-IDF vector of each document in the corpus
    tfidf_corpus = []
    for doc in corpus:
        tf = defaultdict(int)
        words = doc.split()
        for word in words:
            if word in vocab:
                tf[word] += 1
        tfidf = {}
        for word in set(words):
            if word in vocab:
                tfidf[word] = tf[word] * idf[word]
        tfidf_corpus.append(tfidf)

    return tfidf_corpus

def get_features(train_file, test_file, dimension=10000):
    '''
    extract features and labels from training and testing files
    demension(int): The number of the demension of the feature
    '''
    print("Building dictionary now")
    train_items, train_labels = read_csv(train_file)
    test_items, test_labels = read_csv(test_file)

    vocab = build_vocab(train_items)[:dimension]
    print("Dictionary is done now")
    word2id = {}
    for id, item in enumerate(vocab):
        word2id[item[0]] = id
    train_tfidf = compute_tfidf(train_items, word2id.keys())
    test_tfidf = compute_tfidf(test_items, word2id.keys())
    train_features = [{word2id[word]: tfidf for word, tfidf in doc.items()} for doc in train_tfidf]
    test_features = [{word2id[word]: tfidf for word, tfidf in doc.items()} for doc in test_tfidf]
    return train_features, train_labels, test_features, test_labels, word2id

def get_shuffle(list1,list2):
    '''
    Get training samples and labels shuffled
    '''
    merge_list = [[list1[i],list2[i]] for i in range(len(list1))]
    random.shuffle(merge_list)
    list1 = [merge_list[0] for i in range(len(list1))]
    list2 = [merge_list[1] for i in range(len(list2))]
    return list1,list2



