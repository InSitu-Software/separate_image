from unittest import TestCase
import numpy as np

import separate_image


class TestSplit(TestCase):
    def test_split_by_lines_rows(self):

        image = np.zeros((200, 100))
        for r in range(0,200,20):
            image[r,:] = 255

        image = np.matrix(image)

        splitting = separate_image.split(
            image,
            'blackline',
            'rows'
        )

        self.assertTrue(len(splitting) == 11)

    def test_split_by_whitesapce_rows(self):

        image = np.ones((200, 100))*255
        for r in range(20,200,20):
            image[r:r+2,:] = 0

        image = np.matrix(image)

        splitting = separate_image.split(
            image,
            'whitespace',
            'rows'
        )

        self.assertTrue(len(splitting) == 10)

    def test_split_by_lines_columns(self):

        image = np.zeros((200, 100))
        for r in range(0,200,20):
            image[r,:] = 255

        image = np.matrix(image).T

        splitting = separate_image.split(
            image,
            'blackline',
            'columns'
        )

        self.assertTrue(len(splitting) == 11)

    def test_split_by_whitesapce_columns(self):

        image = np.ones((200, 100))*255
        for r in range(20,200,20):
            image[r:r+2,:] = 0

        image = np.matrix(image).T

        splitting = separate_image.split(
            image,
            'whitespace',
            'columns'
        )

        self.assertTrue(len(splitting) == 10)