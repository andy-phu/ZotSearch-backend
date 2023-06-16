from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import json
from bs4 import BeautifulSoup
from urllib.parse import urldefrag
import os
import math
import re

ALPHANUMERIC_WORDS = re.compile('[a-zA-Z0-9]+')


class Posting:
    def __init__(self, id, frequency):
        self._id = id
        self._frequency = frequency

    @property
    def id(self):
        return self._id

    @property
    def frequency(self):
        return self._frequency


'''
Code to parse JSON file and tokenize words alphanumerically, ignoring stopwords. Will also fix broken HTML
'''


def parseFile(filePath: str):
    # filePath is a path to a JSON object file. Get the URL and content from obj file.
    with open(filePath, 'r') as file:
        json_obj = json.load(file)

        url = json_obj['url']

        # Defragging the URL
        url, fragment = urldefrag(url)

        # Using beautifulsoup to parse HTML content
        page_obj = BeautifulSoup(json_obj['content'], 'lxml')

        # Tokenizing the text and storing in dictionary. Key (token) value (frequency)
        return (pageTokenize(page_obj), url)


def pageTokenize(page: object):
    '''
    Tokenizes the content retrieved from BeautifulSoup's get_text().
    Returns a dictionary of the tokens as keys and frequency as values.
    This tokenizer also takes the *stems* from every token and stores it as
    keys.
    '''

    # Tokenizing the page and storing tokens in a list
    # regTokenizer = RegexpTokenizer(r'\w+')
    # tokens = regTokenizer.tokenize(page.get_text())

    text = page.get_text()

    # Stemming each token and adding to a dictionary
    stemmer = PorterStemmer()
    stems = dict()
    for token in re.findall(ALPHANUMERIC_WORDS, text):
        stemmedWord = stemmer.stem(token)

        # Checking if it's already in the dictioanry - if it is, add by 1. If not, add a new entry
        if stemmedWord in stems:
            stems[stemmedWord] += 1
        else:
            stems[stemmedWord] = 1

    return stems


def calculateTokenScore(frequency, numDocumentsCorpus, numDocumentsTerm):
    '''
    Calculates the TF-IDF score for a single token and the amount of times this word appears in some document.
    Returns the TF-IDF score of the combination of these two.

    @params - frequency: int, numDocumentsCorpus: int, numDocumentsTerm: int
    @returns - score: float
    '''

    # First, calculating the weighting of the frequencycd des
    tf = 0

    if frequency > 0:
        tf = 1 + math.log10(frequency)

    # Calculate the idf
    idf = math.log10(numDocumentsCorpus / numDocumentsTerm)
    return tf * idf


def createInvertedIndex():
    # Data structure (dictionary) to hold the inverted index in memory
    index = dict()

    # Dictionary to hold the mapping between page ID and url
    pageIDs = dict()

    pageCounter = 0

    # Iterating through all the inner folders in DEV folder
    path_to_inner = '../FinalSubmission/DEV/'
    for folder in os.listdir(path_to_inner):
        if folder == '.DS_Store':
            continue
        # Iterating through all the JSON files in the inner folder
        json_files = []

        for filename in os.listdir(os.path.join(path_to_inner, folder)):
            json_files.append(os.path.join(path_to_inner, folder, filename))

        for json_file in json_files:

            # Processing each json file
            words, url = parseFile(json_file)

            for word, counter in words.items():

                # Creating a posting
                post = Posting(pageCounter, counter)

                # Assigning to dictionary
                if word in index:
                    index[word].append(post)
                else:
                    index[word] = []
                    index[word].append(post)

            # After processing, store a mapping between the actual file and the id
            pageIDs[pageCounter] = url
            pageCounter += 1

    # Once finished, put output to file
    with open('invertedIndex.txt', 'w', encoding='utf-8') as f:
        new_index = sorted(index.items())

        for key, value in new_index:

            # Calculate the idf
            idf = math.log10(len(pageIDs) / len(value))

            f.write(f"{key},{idf},{len(value)}")

            for post in value:
                # Calculating the TF-IDF Score and replacing the word frequency.
                # score = calculateTokenScore(post.frequency, len(pageIDs), len(value))

                f.write(f",{post.id}:{post.frequency}")
            f.write("\n")

    with open('mapping.txt', 'w', encoding='utf-8') as f:
        for key, value in pageIDs.items():
            f.write(f"{key} {value}\n")


if __name__ == '__main__':
    createInvertedIndex()