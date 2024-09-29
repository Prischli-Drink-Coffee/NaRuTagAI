import random
# import synonyms
from pymorphy2 import MorphAnalyzer
from nltk.tokenize import word_tokenize
import re

import nltk

nltk.download('punkt_tab')

morph = MorphAnalyzer()


class Augmentor:
    """
    Augments a single text string in Russian into three different variations.

    Parameters:
    text (str): The input text string in Russian.

    Returns:
    tuple of str: A tuple containing three augmented versions of the input text.
    """

    def __init__(self):
        pass

    # def synonym_replacement(self):
    #     """
    #     Replaces words with their synonyms.
    #     """
    #     words = word_tokenize(self.text, language='russian')
    #     augmented_words = []
    #     for word in words:
    #         word_lemma = morph.parse(word)[0].normal_form
    #         synonyms_list = synonyms.get_synonyms(word_lemma, lang='ru')
    #         if synonyms_list:
    #             synonym = random.choice(synonyms_list)
    #             augmented_words.append(synonym)
    #         else:
    #             augmented_words.append(word)
    #     return ' '.join(augmented_words)


    def word_permutation(self, text):
        """
        Randomly permutes the order of words.
        """
        words = word_tokenize(text, language='russian')
        index1 = random.randint(0, len(words) - 2)
        index2 = random.choice([index1 - 1, index1 + 1])
        words[index1], words[index2] = words[index2], words[index1]
        return ' '.join(words)

    def random_replacement(self, text):  # TODO несколько ошибок
        words = word_tokenize(text, language='russian')
        index = random.randint(0, len(words) - 1)  # выбираем рандомное слово, в котором будем делать ошибку
        word = words[index]

        replacement = random.choice('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        index_letter = random.randint(0, len(word) - 1)
        word = word[:index] + replacement + word[index + 1:]

        words = words[:index] + [word] + words[index + 1:]
        return ' '.join(words)

    def random_delete(self, text):
        words = word_tokenize(text, language='russian')
        index = random.randint(0, len(words) - 1)  # выбираем рандомное слово, в котором будем делать ошибку
        word = words[index]

        index_letter = random.randint(0, len(word) - 1)
        word = word[:index] + word[index + 1:]

        words = words[:index] + [word] + words[index + 1:]
        return ' '.join(words)

    def augment(self, text):
        # Generate the three variations
        text = self.word_permutation(text)
        text = self.random_replacement(text)
        text = self.random_delete(text)

        return text


# Пример использования функции
text = "мы пытаемся сделать что-то нормальное но выходит как обычно и я сейчас просто расплачусь"
augmented_versions = Augmentor()

print("Оригинальный текст:", text)
for i in range(10):
    d = augmented_versions.augment(text)
    print("Аугментированный вариант ", d)