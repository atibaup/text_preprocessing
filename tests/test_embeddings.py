import unittest
from text_preprocessing import embeddings
import numpy as np

class TestEmbeddings(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(
            embeddings.tokenize('this is my text'),
            ['this', 'is', 'my', 'text']
        )

    def test_bow_embed_on_empty_texts(self):
        expected = np.zeros((1, len(embeddings.VOCABULARY)))
        expected[0, embeddings.VOCABULARY.index(embeddings.EMPTY_TOKEN)] = 1
        self.assertTrue(np.allclose(
            embeddings.bow_embed(['']),
            expected
        ))

    def test_bow_embed_on_single_words(self):
        expected = np.zeros((1, len(embeddings.VOCABULARY)))
        expected[0, embeddings.VOCABULARY.index(embeddings.OTHER_TOKEN)] = 1
        self.assertTrue(np.allclose(
            embeddings.bow_embed(['this']),
            expected
        ))

    def test_bow_embed_on_multiple_other_words(self):
        expected = np.zeros((1, len(embeddings.VOCABULARY)))
        expected[0, embeddings.VOCABULARY.index(embeddings.OTHER_TOKEN)] = 5
        self.assertTrue(np.allclose(
            embeddings.bow_embed(['this is my test text']),
            expected
        ))

    def test_bow_embed_on_multiple_words(self):
        expected = np.zeros((1, len(embeddings.VOCABULARY)))
        expected[0, 0] = 1
        expected[0, 1] = 1
        expected[0, 2] = 1
        expected[0, 3] = 1
        self.assertTrue(np.allclose(
            embeddings.bow_embed(['python java sql delphi']),
            expected
        ))

class TestHashEmbeddings(unittest.TestCase):
    def test_constructor_works(self):
        hash_embedder = embeddings.HashEmbedder(100)
        self.assertEqual(hash_embedder.dim, 100)

    def test_embed_one_returns_correct_shape(self):
        hash_embedder = embeddings.HashEmbedder(3)
        self.assertEqual(
            hash_embedder.embed_one('this is my test text').shape,
            (3,)
        )

    def test_embed_returns_correct_shape(self):
        hash_embedder = embeddings.HashEmbedder(3)
        self.assertEqual(
            hash_embedder.embed(['this is my test text']).shape,
            (1, 3)
        )

    def test_embed_on_one_dimension(self):
        hash_embedder = embeddings.HashEmbedder(1)
        print(hash_embedder.embed(['this is my test text']))
        self.assertTrue(np.allclose(
            hash_embedder.embed(['this is my test text']),
            np.array([[5.]])
        ))

    # example of "regression" type test
    def test_embed_on_long_test(self):
        hash_embedder = embeddings.HashEmbedder(30)
        example = ['this is a long test test text that I want to use to prevent regressions']
        expected = np.array([[0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0,
                              0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0,
                              0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0,
                              1.0, 0.0, 0.0, 0.0, 0.0, 2.0]])
        self.assertTrue(np.allclose(
            hash_embedder.embed(example),
            expected
        ))

# Using `parameterized`

from parameterized import parameterized
import random

def random_doc_generator(n, k):
    for _ in range(n):
        yield [' '.join(random.choices(embeddings.VOCABULARY, k=k))], k

class TestEmbeddingsParameterized(unittest.TestCase):
    @parameterized.expand([
        ('this is my text', ['this', 'is', 'my', 'text']),
        ('this', ['this']),
        ('', [])
    ])
    def test_tokenize(self, input, expected_output):
        self.assertEqual(
            embeddings.tokenize(input),
            expected_output
        )

    # using a list
    # it shouldn't work for __EMPTY_, that's why we exclude it from
    # test cases
    @parameterized.expand(embeddings.VOCABULARY[:-1])
    def test_bow_embed_on_single_words(self, input):
        expected = np.zeros((1, len(embeddings.VOCABULARY)))
        expected[0, embeddings.VOCABULARY.index(input)] = 1
        self.assertTrue(np.allclose(
            embeddings.bow_embed([input]),
            expected
        ))

    # using a generator to test an invariance
    @parameterized.expand(random_doc_generator(10, 10))
    def test_bow_embed_on_multiple_words_num_non_zero(self, input, expected_sum):
        self.assertEqual(
            embeddings.bow_embed(input).sum(),
            expected_sum
        )

if __name__ == '__main__':
    unittest.main()