from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer


text = "Hello #StopAbuseNow! This is STUPID!!! @user :-)"

text_processor = TextPreProcessor(
    normalize=[
        "url",
        "email",
        "percent",
        "money",
        "phone",
        "user",
        "time",
        "date",
        "number"
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


tokens = text_processor.pre_process_doc(text)

print("Original:")
print(text)

print("\nEkphrasis output:")
print(tokens)

print("\nAs string:")
print(" ".join(tokens))