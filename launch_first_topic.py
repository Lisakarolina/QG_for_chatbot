from pipelines import pipeline
import wikipedia
from nltk import sent_tokenize

FILE = "first_topic.txt"   # insert the document to direct the results to

nlp = pipeline("multitask-qa-qg")

# nlp = pipeline("question-generation", model="valhalla/t5-small-qg-prepend",
#                qg_format="prepend")


def prepare():
    """segment the text into chunks that meet the max_length criterium"""
    text = wikipedia.page("Cat").content
    index = 252
    result = []

    sentences = sent_tokenize(text)  # segment text into list of sentences

    position = 0
    container = []

    for sentence in sentences:
        if(position <= index):
            number_words = len(sentence.split())
            number_tokens = number_words + number_words-1
            container.append(sentence)
            position += number_tokens
            continue
        container = " ".join(container)
        result.append(container)
        position = 0
        container = []

    return(result)


result = []
for element in prepare():
    result.append(nlp(element))

with open(FILE, "w") as f:
    for elem in result:
        for ele in elem:
            f.write(str(ele))
            f.write("\n")
