import pandas as pd
import re
import nltk
from spacy.tokenizer import Tokenizer
from spacy.lang.ru import Russian
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
from io import StringIO
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)


class FiltrationAlgorithm:
    def __init__(self):
        """
        Initializes the FiltrationAlgorithm class.

        Sets up the Spacy tokenizer for Russian language and loads stopwords for
        English and Russian languages.
        """
        self.nlp = Russian()
        self.tokenizer = Tokenizer(self.nlp.vocab)
        self.english_stopwords = set(stopwords.words('english'))
        self.russian_stopwords = set(stopwords.words('russian'))

    def calculate_symbol_statistics(self, dataframe, statistics_df, name):
        """
        Calculates average symbol statistics for 'title' and 'description' columns
        and updates the statistics DataFrame.

        Parameters:
        dataframe (pd.DataFrame): DataFrame containing 'title' and 'description' columns.
        statistics_df (pd.DataFrame): DataFrame to store calculated statistics.
        name (str): The name of the current dataset or processing step.

        Returns:
        pd.DataFrame: Updated statistics DataFrame including the new statistics.
        """
        if dataframe.empty:
            return statistics_df

        total_title_length = 0
        total_description_length = 0

        for i in range(len(dataframe)):
            total_title_length += len(dataframe['title'][i])
            total_description_length += len(dataframe['description'][i])

        avg_title_length = total_title_length / len(dataframe)
        avg_description_length = total_description_length / len(dataframe)

        new_stats = pd.DataFrame({
            "name": [name],
            "average_title_length": [avg_title_length],
            "average_description_length": [avg_description_length]
        })

        statistics_df = pd.concat([statistics_df, new_stats], ignore_index=True)
        return statistics_df

    def links_extraction(self, text):
        """
        Removes URLs from the input text.

        Parameters:
        text (str): Input text from which URLs need to be removed.

        Returns:
        str: Text with URLs removed.
        """
        text = text.lower()
        return re.sub(r'http\S+|https\S+|www.\S+', '', text)

    def special_symbols_extraction(self, text):
        """
        Removes special symbols and HTML entities from the input text.

        Parameters:
        text (str): Input text from which special symbols need to be removed.

        Returns:
        str: Text with special symbols removed.
        """
        special_symbols = (
            "&quot;", "&apos;", "&lt;", "&gt;", "&amp;",
            "\n", "\r", "\t", "\\\\", "\\\"", "\\'",
            "<br>", "<i>", "<b>", "<u>", "%20", "%2F", "%3A",
            "&nbsp;", "&copy;", "&reg;", "&euro;", "&yen;", "&pound;", "&dollar;"
        )
        for symbol in special_symbols:
            text = text.replace(symbol, "")
        text = re.sub(r'#&.*$', '', text)
        return text

    def punctuation_extraction(self, text):
        """
        Removes punctuation from the input text.

        Parameters:
        text (str): Input text from which punctuation needs to be removed.

        Returns:
        str: Text with punctuation removed.
        """
        punkts = (",", ".", "!", "?", ";", ":", "-", "_", "(", ")", "[", "]", "{", "}", "\'", "/", "|", "'")
        for symbol in punkts:
            text = text.replace(symbol, '')
        return text

    def remove_stopwords(self, text):
        """
        Removes stopwords from the input text.

        Parameters:
        text (str): Input text from which stopwords need to be removed.

        Returns:
        str: Text with stopwords removed.
        """
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in self.russian_stopwords]
        return ' '.join(filtered_words)

    def process_data(self, dataframe, statistics_df, checkpoint_file, start_step=0):
        """
        Processes the DataFrame through various text filtering steps and updates statistics.
        Saves checkpoint progress to resume processing from a specific step.

        Parameters:
        dataframe (pd.DataFrame): DataFrame containing 'title' and 'description' columns to process.
        statistics_df (pd.DataFrame): DataFrame to store the calculated statistics.
        checkpoint_file (str): File to save the progress checkpoint.
        start_step (int, optional): Step index to start processing from. Defaults to 0.

        Returns:
        pd.DataFrame: Processed DataFrame.
        pd.DataFrame: Updated statistics DataFrame.
        """
        steps = [
            ('original', None, None),
            ('URL-extraction', self.links_extraction, None),
            ('Special symbols', self.special_symbols_extraction, None),
            ('Punctuation', self.punctuation_extraction, None),
            ('Stopwords', self.remove_stopwords, None)
        ]

        start_step = min(start_step, len(steps))

        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = f.read().strip()
                if checkpoint:
                    start_step = int(checkpoint)

        for idx, (step_name, func, post_func) in enumerate(steps):
            if idx < start_step:
                continue

            if func:
                dataframe['title'] = dataframe['title'].apply(func)
                dataframe['description'] = dataframe['description'].apply(func)

            statistics_df = self.calculate_symbol_statistics(dataframe, statistics_df, step_name)
            statistics_df.to_csv('statistics.csv', index=False)

            with open(checkpoint_file, 'w') as f:
                f.write(str(idx + 1))

        return dataframe, statistics_df

    def load_checkpoint(self, checkpoint_file):
        """
        Loads and returns the current checkpoint step from the checkpoint file.

        Parameters:
        checkpoint_file (str): File to read the checkpoint from.

        Returns:
        int: The checkpoint step index or 0 if no checkpoint file exists.
        """
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = f.read().strip()
                if checkpoint:
                    return int(checkpoint)
        return 0


# # Usage example:
# filtration = FiltrationAlgorithm()
#
# df = pd.read_csv('metadata.csv')
# df= df[['title', 'description']].copy()
# df['title'] = df['title'].fillna('')
# df['description'] = df['description'].fillna('')
#
# statistics_df = pd.DataFrame()
# checkpoint_file = 'checkpoint.txt'
# start_step = filtration.load_checkpoint(checkpoint_file)
# df, statistics_df = filtration.process_data(df, statistics_df, checkpoint_file, start_step)
#
# print(statistics_df)
