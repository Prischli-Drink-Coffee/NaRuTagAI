from src.modelling.filtration_algorithm import FiltrationAlgorithm
from textattack.augmentation import WordNetAugmenter
from nltk.stem import LancasterStemmer

filtration = FiltrationAlgorithm()
augmenter = WordNetAugmenter()
stemmer = LancasterStemmer()


def process_single_text(text, use_augmentation=False, use_lemmatization=False):
    """
    Processes a single text string using the FiltrationAlgorithm class methods.

    Parameters:
    text (str): The text to be processed.

    Returns:
    str: The processed text.
    """

    steps = [
        ('URL-extraction', filtration.links_extraction),
        ('Special symbols', filtration.special_symbols_extraction),
        ('Punctuation', filtration.punctuation_extraction),
        ('Stopwords', filtration.remove_stopwords)
    ]

    for idx, (step_name, func) in enumerate(steps):
        if func:
            text = func(text)

    # Аугментация текста, если флаг use_augmentation активен
    if use_augmentation:
        text = augmenter.augment(text)[0]

    if use_lemmatization:
        text = stemmer.stem(text)

    return text
