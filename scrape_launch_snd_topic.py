from pipelines import pipeline
from nltk import sent_tokenize
import requests
from bs4 import BeautifulSoup

FILE = "second_topic.txt"    # the resulting qa-pairs go here

page = requests.get("https://www.thespruce.com/grow-weeping-fig-indoors-1902440")
content = page.content
soup = BeautifulSoup(content, 'lxml')

# extract the table first:
rows = ""
table = soup.find("table", class_="mntl-sc-block-table__table")
rows = table.find_all("tr")

cleaned_text = ""
for row in rows:
    row = row.find_all("td")
    cleaned_text += f"{row[0].text}: {row[1].text}. "  # transfer table cells in text format

# extract text passages:
passages = soup.find_all(class_='comp mntl-sc-block mntl-sc-block-html')
for passage in passages:
    cleaned_text += passage.text


nlp = pipeline("multitask-qa-qg")

# nlp = pipeline("question-generation", model="valhalla/t5-small-qg-prepend",
#                qg_format="prepend")


def prepare():
    """segment the text into chunks that meet the max_length criterium"""
    index = 252
    result = []

    sentences = sent_tokenize(cleaned_text)  # segment text into list of sentences

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
