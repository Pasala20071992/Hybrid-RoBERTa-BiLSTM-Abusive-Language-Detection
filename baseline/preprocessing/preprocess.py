from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from nltk.stem import SnowballStemmer


# Snowball Stemmer
stemmer = SnowballStemmer("english")


# Ekphrasis preprocessor
text_processor = TextPreProcessor(
    normalize=[
        "url",
        "email",
        "percent",
        "money",
        "phone",
        "time",
        "date",
        "number",
        "user"
    ],

    annotate={
        "hashtag",
        "allcaps",
        "elongated",
        "repeated"
    },

    segmenter="twitter",
    corrector="twitter",

    unpack_hashtags=True,
    unpack_contractions=True,

    spell_correct_elong=False,

    tokenizer=SocialTokenizer(lowercase=True).tokenize
)


def preprocess_text(text):

    # Convert to string
    text = str(text)

    # -----------------------------------
    # 1. Ekphrasis preprocessing
    # -----------------------------------

    tokens = text_processor.pre_process_doc(text)

    # -----------------------------------
    # 2. Convert tokens to string
    # -----------------------------------

    text = " ".join(tokens)

    # -----------------------------------
    # 3. Remove special characters
    # -----------------------------------

    text = text.replace("@", "")
    text = text.replace("&", "")
    text = text.replace(":", "")
    text = text.replace(";", "")
    text = text.replace(",", "")
    text = text.replace("%", "")
    text = text.replace("#", "")

    # -----------------------------------
    # 4. Remove extra spaces
    # -----------------------------------

    text = " ".join(text.split())

    # -----------------------------------
    # 5. Snowball stemming
    # -----------------------------------

    words = text.split()

    stemmed_words = []

    for word in words:

        # Don't stem special tokens
        if word.startswith("<") and word.endswith(">"):
            stemmed_words.append(word)

        else:
            stemmed_words.append(stemmer.stem(word))

    text = " ".join(stemmed_words)

    return text


if __name__ == "__main__":

    text = "Hello #StopAbuseNow #NoMoreHate #CyberBullying"

    result = preprocess_text(text)

    print("Original:")
    print(text)

    print("\nProcessed:")
    print(result)