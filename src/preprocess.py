import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# angielskie stopwords
STOP_WORDS = set(stopwords.words("english"))

# lematyzator
lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    Czyszczenie pojedynczej wiadomości SMS.
    """

    # lowercase
    text = text.lower()

    # usunięcie URL
    text = re.sub(r"http\S+|www\S+", " ", text)

    # pozostawienie tylko liter i spacji
    text = re.sub(r"[^a-z\s]", " ", text)

    # tokenizacja
    tokens = word_tokenize(text)

    # stopwords + lematyzacja
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]

    return " ".join(tokens)