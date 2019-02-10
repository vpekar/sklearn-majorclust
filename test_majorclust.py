
import numpy as np
from unittest import TestCase
from sklearn.feature_extraction.text import TfidfVectorizer

import majorclust


class TestMajorclust(TestCase):

    def setUp(self):
        pass

    def test_get_sim_matrix(self):
        X = np.array([
            [1, 2, 3, 4],
            [1, 2, 3, 4],
            [1, 2, 5, 6],
        ])
        sim_matrix = majorclust.get_sim_matrix(X, 0.0)
        self.assertEqual(sim_matrix.shape, (3, 3))
        self.assertAlmostEqual(sim_matrix[0, 1], 1.0, 5)

    def test_majorclust(self):
        sim_matrix = np.array([[0, 1, 0, 0],
                               [1, 0, 0, 0],
                               [0, 0, 0, 1],
                               [0, 0, 1, 0]])
        labels = majorclust.majorclust(sim_matrix)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])

    def test_text_clustering(self):
        texts = [
            "foo blub baz",
            "foo bar baz",
            "asdf bsdf csdf",
            "foo bab blub",
            "csdf hddf kjtz",
            "123 456 890",
            "321 890 456 foo",
            "123 890 uiop",
        ]
        mc = majorclust.MajorClust(sim_threshold=0.0)
        X = TfidfVectorizer().fit_transform(texts)
        mc.fit(X)
        # cluster 1: 0, 1, 3
        self.assertTrue(mc.labels_[0] == mc.labels_[1] == mc.labels_[3])
        # cluster 2: 2, 4
        self.assertTrue(mc.labels_[2] == mc.labels_[4])
        # cluster 3: 5, 6, 7
        self.assertTrue(mc.labels_[5] == mc.labels_[6] == mc.labels_[7])
