import numpy as np
import hashlib

OTHER_TOKEN = "__OTHER__"
EMPTY_TOKEN = "__EMPTY__"
VOCABULARY = ['python', 'java', 'sql', 'delphi', 'c++', OTHER_TOKEN, EMPTY_TOKEN]

def tokenize(doc):
    return doc.split()

def bow_embed(documents):
    embedded_docs = []
    for document in documents:
        tokenized_doc = tokenize(document)
        embedded_doc = np.zeros(len(VOCABULARY))
        for token in tokenized_doc:
            if token in VOCABULARY:
                embedded_doc[VOCABULARY.index(token)] = 1
            else:
                embedded_doc[VOCABULARY.index(OTHER_TOKEN)] = 1
        if np.alltrue(embedded_doc == 0.0):
            embedded_doc[VOCABULARY.index(EMPTY_TOKEN)] = 1
        embedded_docs.append(embedded_doc)
    return np.array(embedded_docs)

def hash_token(token):
    # this just maps any token into an integer
    return int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)

class HashEmbedder:
    def __init__(self, dim):
        self.dim = dim
    def embed(self, texts):
        return np.array([self.embed_one(text) for text in texts])
    def embed_one(self, document):
        embedded = np.zeros(self.dim)
        indices = [hash_token(token) % self.dim for token in tokenize(document)]
        embedded[indices] = 1
        return embedded