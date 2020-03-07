# text_preprocessing
----------------------

This is a project to practice some of the things
 we've learned on git, env management, makefiles,
 templates and python during the [2020 ML in Prod training](mlinproduction.github.io). 

We will build a library (package) to perform a different type of "embedding" (conversion
from text to arrays), and integrate it into the capstone project. While we do it, we will 
practice with many of the tools we have been learning.

1. Create a `text_preprocessing` package using an existing cookiecutter template
or manually, with the following structure and empty files:

```bash
text_processing/
... text_processing/
....... __init__.py
... tests/
....... .gitkeep
... setup.py 
... README.md
```

2. Go to the new package folder and create a git repository:

```
cd text_processing
git init .
git add .
git status
# check everything is there :)
git commit -m "Initial skeleton."
```

3. Go to github and create a git repository `text_preprocessing` 
(**WARNING**: do not create a README.md or .gitinore file)

4. Add the remote to your local git repository

git remote add origin https://github.com/atibaup/text_preprocessing.git
git push -u origin master

5. Let's set up a conda environment for this project:

`conda create -n text_preprocessing Python=3.7`

6. We are now ready to start coding. We will create a `bow_embed` function in an embeddings.py module:

```python
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
```

7. We will now add unit tests to make sure this function is running well

```python
import unittest

class TestEmbedding(unittest.TestCase):
	def test_bow_embed_on_empty_texts(self):
		pass

	def test_bow_embed_on_single_words(self):
		pass
		
if __name__ == '__main__':
    unittest.main()
```

> Advanced 1: use parameterized

8. When we are happy and our tests pass, we will commit the changes 

9. We will now add our package as a dependency of the `train` package in the capstone project

10. We will make the changes in `train.py` to now use and test our new `bow_embedding`

from CLI: 

> pip install git+https://github.com/django/django.git@45dfb3641aa4d9828a7c5448d11aa67c7cbd7966#egg=django[argon2]

in conda env, add those lines to the conda env yml:

> - pip:
>     - "--editable=git+https://github.com/pythonforfacebook/facebook-sdk.git@8c0d34291aaafec00e02eaa71cc2a242790a0fcc#egg=facebook_sdk-master"


[Bonus]

11. Our `bow_embedding` model was a little silly, because it had a 10 word vocabulary, hence our accuracy was not very good. We are now going to improve it by implementing
a HashEmbedding, which in this case it will need to be a class because it will hold state:


class HashEmbedding:
	def __init__(self, dim):
		self.dim = dim
	def embed(self, texts):
		return [self.embed_one(text) for text in texts]

12. Let's add 

