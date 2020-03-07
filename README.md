# text_preprocessingPracticing everything we've learned on git, env management, makefiles, templates and python
------------------------------------------------------------------------------------------

We will build a library (package) to perform a different type of "embedding" (conversion
from text to arrays), and integrate it into the capstone project. While we do it, we will 
practice with many of the tools we have been learning.

1. Create a `text_preprocessing` package using the `XXX` template

text_processing/
... text_processing/
....... __init__.py
... tests/
....... .gitkeep
... setup.py 

2. Go to the new package folder and create a git repository:

cd text_processing
git init .
git add .
git status
git add tests/.gitkeep

git commit -m "Initial skeleton."

3. Go to github and create a git repository `text_preprocessing`

4. Add the remote to your local git repository

git remote add origin https://github.com/atibaup/text_preprocessing.git
git push -u origin master

git clone https://github.com/atibaup/text_preprocessing.git text_preprocessing_2
cd text_preprocessing
rm -Rf .git
cp -R text_preprocessing/* text_preprocessing_2

git add .
git status
git commit -m "Initial skeleton."
git push origin master

5. Let's set up a conda environment for this project:

conda create -n text_preprocessing Python=3.7

6. We are now ready to start coding. We will create a `bow_embed` function in an embeddings.py module:


def bow_embed(documents):
	"""
	takes a list of documents (lists of strings) and returns
	a numpy array of shape n_documents x n_tokens
	"""

VOCABULARY = ['python', 'java', 'sql', 'delphi', 'c++', "__OTHER__", "__EMPTY__"]

def bow_embed(texts):
	# returns a bag of word representation of each line in texts
	# 1. tokenize
	# 2. binarize
	# 3. map to an array

7. We will now add unit tests to make sure this function is running well

class TestEmbedding(unittest.TestCase):
	def test_bow_embed_on_empty_texts(self):
		pass

	def test_bow_embed_on_single_words(self):
		pass

> Advanced 1: use parameterized

8. When we are happy and our tests pass, we will commit the changes 

9. We will now add our package as a dependency of the `train` package in the capstone project

10. We will make the changes in `train.py` to now use and test our new `bow_embedding`

from CLI: 
pip install git+https://github.com/django/django.git@45dfb3641aa4d9828a7c5448d11aa67c7cbd7966#egg=django[argon2]

in conda env:

- pip:
     - "--editable=git+https://github.com/pythonforfacebook/facebook-sdk.git@8c0d34291aaafec00e02eaa71cc2a242790a0fcc#egg=facebook_sdk-master"


[Bonus]

11. Our `bow_embedding` model was a little silly, because it had a 10 word vocabulary, hence our accuracy was not very good. We are now going to improve it by implementing
a HashEmbedding, which in this case it will need to be a class because it will hold state:


class HashEmbedding:
	def __init__(self, dim):
		self.dim = dim
	def embed(self, texts):
		return [self.embed_one(text) for text in texts]

12. Let's add 

