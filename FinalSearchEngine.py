# Search Engine
import re
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from invertedIndexCreatorBasic import Posting
import math
import itertools
import time
import heapq
import numpy as np

'''
Engine will show the top 5 results (no ranking, because boolean) for each result.
Full results will be output into a text file. 
'''


class SearchEngine:
    def __init__(self, s, regExp, invertedIndexFile, urlMappingsFile, indexOfIndex=None, lengthOfDoc=None,
                 boldWordsFile=None, boldIndexOfIndex=None, headerWordsFile=None, headerIndexOfIndex=None,
                 titleWordsFile=None, titleIndexOfIndex=None, zoneInvertedIndex = None, zoneIndexOfIndex = None):
        self._stemmer = s  # Stemmer to use
        self._reg = regExp  # Regular expression to use

        self._invertedIndexFile = invertedIndexFile  # The OPEN FILE DESCRIPTOR of the inverted index.
        self._urlMappingsFile = urlMappingsFile  # The OPEN FILE DESCRIPTOR of the mapping file.
        self._lengthofDoc = lengthOfDoc
        self._boldIndexFIle = boldWordsFile
        self._headerIndexFIle = headerWordsFile
        self._titleIndexFIle = titleWordsFile
        self._zoneInvertedIndex = zoneInvertedIndex

        self._urlMappings = self.loadUrlMappings()  # Loads the url mappings file into a dictionary
        self._indexOfIndex = self.loadIndexOfIndex(indexOfIndex)
        self._docLengths = self.loadDocLengths()
        self._boldIndex = self.loadBoldIndexOfIndex(boldIndexOfIndex)
        self._headerIndex = self.loadheaderIndexOfIndex(headerIndexOfIndex)
        self._titleIndex = self.loadtitleIndexOfIndex(titleIndexOfIndex)
        #self._zoneIndex = self.loadzoneIndexOfIndex(zoneIndexOfIndex)   

    '''
    creates the mapping and goes line by line on the urlMappings 
    file and seperates the lines on the url mapping files to create a map out of it 
    '''
    def loadUrlMappings(self): 
        mappings = dict()
        for line in self._urlMappingsFile:
            id, url = line.split()
            mappings[id] = url

        return mappings


    '''
    gets the doc lengths and creates new entries in dictionaries using them 
    '''
    def loadDocLengths(self):
        mappings = dict()
        for line in self._lengthofDoc:
            items = line.split(',')
            key = items[0] #items[0] is the docID
            values = [value.strip() for value in items[1:]]
            mappings[key] = values

        return mappings

    def loadIndexOfIndex(self, indexOfIndex: str):
        '''
        Opens the index of index file specified by parameter, parses the
        file, returns a dictionary of the word and position, and closes file.
        '''
        returnable = dict()
        with open(indexOfIndex, 'r') as f:
            for line in f:
                entry = line.split()
                returnable[entry[0]] = entry[1]

        return returnable

    '''
    gets the index of index and goes line by line for it makes the returnable have key: token
    value: the frequncy, posting list for given token
    '''
    # def loadzoneIndexOfIndex(self, indexOfIndex: str):
    #     returnable = dict()
    #     with open(indexOfIndex, 'r') as f: 
    #         for line in f: #each line in indexOfIndex has the token and the position of it in the file 
    #             entry = line.strip().split(',') 
    #             key = entry[0]
    #             value = ','.join(entry[1:])  # Reconstruct the value string
    #             returnable[key] = value

    #     return returnable

    def loadBoldIndexOfIndex(self, indexOfIndex: str):
        '''
        Opens the index of index file specified by parameter, parses the
        file, returns a dictionary of the word and position, and closes file.
        '''
        returnable = dict()
        with open(indexOfIndex, 'r') as f:
            for line in f:
                entry = line.split()
                returnable[entry[0]] = entry[1]

        return returnable

    def loadheaderIndexOfIndex(self, indexOfIndex: str):
        '''
        Opens the index of index file specified by parameter, parses the
        file, returns a dictionary of the word and position, and closes file.
        '''
        returnable = dict()
        with open(indexOfIndex, 'r') as f:
            for line in f:
                entry = line.split()
                returnable[entry[0]] = entry[1]

        return returnable

    def loadtitleIndexOfIndex(self, indexOfIndex: str):
        '''
        Opens the index of index file specified by parameter, parses the
        file, returns a dictionary of the word and position, and closes file.
        '''
        returnable = dict()
        with open(indexOfIndex, 'r') as f:
            for line in f:
                entry = line.split()
                returnable[entry[0]] = entry[1]

        return returnable

    '''
    checks if a docID has a skip ptr and can skip ahead 
    '''
    def hasSkip(self, vector, index):
        try:
            hasSkip = int(vector[index].split(":")[2])
            return True
        except Exception:
            return False

    '''
    if it has a skip assign it to a var and return it, format for posting object = docID:frequency:skipPtr
    '''
    def skip(self, vector, index) -> int:
        hasSkip = int(vector[index].split(":")[2]) #gets the vector[index] and splits based on : then grab the 3rd element and convert to an int 
        return hasSkip

    '''
    finds the index that the skipPtr is pointing to 
    '''
    def skipIndex(self, vector, index) -> int:
        findIndexOfSkip = int(vector[index].split(":")[3])
        return findIndexOfSkip


    def intersect(self, lists):
        if len(lists) < 2:
            x = lists[0].split(",", 1)[0]
            y = lists[0].split(",")[3:]
            resultDict = dict()
            resultDict[x] = y
            return resultDict, len(y)

        # Sort the lists based on the first index in ascending order
        sorted_lists = sorted(lists, key=lambda x: int(x.split(",")[2]))

        answer_length = int(sorted_lists[0].split(",")[2].strip())
        # Initialize the intersection with the first sorted list
        answer = sorted_lists[0].split(",")[3:]  # Split the first sorted list and exclude the first two elements

        first_word = sorted_lists[0].split(",", 1)[0]
        resultDict = dict()
        for i in range(1, len(sorted_lists)):

            current_list_length = int(sorted_lists[i].split(",")[2].strip())
            current_list = sorted_lists[i].split(",")[
                           3:]  # Split the current sorted list and exclude the first two elements
            second_word = sorted_lists[i].split(",", 1)[0]
            intersection = []  # Temporary list to store the intersection of the current list with the answer list

            p1 = 0  # Index for answer list
            p2 = 0  # Index for current list

            while p1 < answer_length and p2 < current_list_length:
                doc_id_1 = int(answer[p1].split(":")[0])  # Extract docID from answer list
                doc_id_2 = int(current_list[p2].split(":")[0])  # Extract docID from current list
                maybeAdd = answer[p1]
                maybeAddThisToo = current_list[p2]
                if doc_id_1 == doc_id_2:
                    if i == 1:
                        if first_word in resultDict:
                            resultDict[first_word].append(maybeAdd)
                        else:
                            resultDict[first_word] = [maybeAdd]

                        if second_word in resultDict:
                            resultDict[second_word].append(maybeAddThisToo)
                        else:
                            resultDict[second_word] = [maybeAddThisToo]

                        intersection.append(doc_id_1)
                        p1 += 1
                        p2 += 1
                    else:
                        if second_word in resultDict:
                            resultDict[second_word].append(maybeAddThisToo)
                        else:
                            resultDict[second_word] = [maybeAddThisToo]

                        intersection.append(doc_id_1)
                        p1 += 1
                        p2 += 1
                elif doc_id_1 < doc_id_2:
                    if self.hasSkip(answer, p1) and (self.skip(answer, p1) <= doc_id_2):
                        while self.hasSkip(answer, p1) and (self.skip(answer, p1) <= doc_id_2):
                            p1 = self.skipIndex(answer, p1)
                    else:
                        p1 += 1
                else:
                    if self.hasSkip(current_list, p2) and (self.skip(current_list, p2) <= doc_id_1):
                        while self.hasSkip(current_list, p2) and (self.skip(current_list, p2) <= doc_id_1):
                            p2 = self.skipIndex(current_list, p2)
                    else:
                        p2 += 1

            answer = ['placeholder', str(len(intersection))] + [f"{element}:0" for element in
                                                                intersection]  # Update the answer list with the intersection  # Update the answer list with the intersection
            answer_length = int(answer[1])
            answer = answer[2:]
        N = len(sorted_lists)

        finished = False  # Flag variable

        while len(answer) < 10 and N > 1 and not finished:
            N -= 1
            for combination in itertools.combinations(sorted_lists, N):
                combined_intersection, length = self.nMinusOneLists(combination)
                if combined_intersection is not None:
                    for key, values in combined_intersection.items():
                        if key in resultDict and len(values) < 10:
                            resultDict[key] = list(set(resultDict[key] + values))
                        else:
                            resultDict[key] = values
                        if len(resultDict[key]) >= 10:  # Check if a list in resultDict has reached 10 elements
                            finished = True
                            break
                if finished:  # Break the outer loop if a list in resultDict has reached 10 elements
                    break
        answer.sort(key=lambda x: int(x.split(':')[0]))
        max_length = 0
        max_value = None

        # Find the maximum length in the dictionary
        for key, values in resultDict.items():
            length = len(values)
            if length > max_length:
                max_length = length
                max_value = values

        # Assign the maximum value to all other values in the dictionary
        for key in resultDict:
            resultDict[key] = max_value

        return resultDict, len(answer)

    def nMinusOneLists(self, sorted_lists):
        if len(sorted_lists) < 2:
            x = sorted_lists[0].split(",", 1)[0]
            y = sorted_lists[0].split(",")[3:]  # Limit the values to the first ten elements
            resultDict = {x: y}
            return resultDict, len(y)

        answer_length = int(sorted_lists[0].split(",")[2].strip())
        # Initialize the intersection with the first sorted list
        answer = sorted_lists[0].split(",")[3:]  # Split the first sorted list and exclude the first two elements

        first_word = sorted_lists[0].split(",", 1)[0]
        resultDict = dict()
        for i in range(1, len(sorted_lists)):

            current_list_length = int(sorted_lists[i].split(",")[2].strip())
            current_list = sorted_lists[i].split(",")[
                           3:]  # Split the current sorted list and exclude the first two elements
            second_word = sorted_lists[i].split(",", 1)[0]
            intersection = []  # Temporary list to store the intersection of the current list with the answer list

            p1 = 0  # Index for answer list
            p2 = 0  # Index for current list

            while p1 < answer_length and p2 < current_list_length:
                doc_id_1 = int(answer[p1].split(":")[0])  # Extract docID from answer list
                doc_id_2 = int(current_list[p2].split(":")[0])  # Extract docID from current list
                maybeAdd = answer[p1]
                maybeAddThisToo = current_list[p2]
                if doc_id_1 == doc_id_2:
                    if i == 1:
                        if first_word in resultDict:
                            resultDict[first_word].append(maybeAdd)
                        else:
                            resultDict[first_word] = [maybeAdd]

                        if second_word in resultDict:
                            resultDict[second_word].append(maybeAddThisToo)
                        else:
                            resultDict[second_word] = [maybeAddThisToo]

                        intersection.append(doc_id_1)
                        p1 += 1
                        p2 += 1
                    else:
                        if second_word in resultDict:
                            resultDict[second_word].append(maybeAddThisToo)
                        else:
                            resultDict[second_word] = [maybeAddThisToo]

                        intersection.append(doc_id_1)
                        p1 += 1
                        p2 += 1
                elif doc_id_1 < doc_id_2:
                    if self.hasSkip(answer, p1) and (self.skip(answer, p1) <= doc_id_2):
                        while self.hasSkip(answer, p1) and (self.skip(answer, p1) <= doc_id_2):
                            p1 = self.skipIndex(answer, p1)
                    else:
                        p1 += 1
                else:
                    if self.hasSkip(current_list, p2) and (self.skip(current_list, p2) <= doc_id_1):
                        while self.hasSkip(current_list, p2) and (self.skip(current_list, p2) <= doc_id_1):
                            p2 = self.skipIndex(current_list, p2)
                    else:
                        p2 += 1

            intersection_length = len(intersection)
            answer = ['placeholder', str(intersection_length)] + [f"{element}:0" for element in
                                                                  intersection]  # Update the answer list with the intersection
            answer_length = intersection_length
            answer = answer[2:]

            if intersection_length >= 10:
                break  # Break out of the loop if the intersection already has 10 or more elements

        return resultDict, len(answer)

    def getUrls(self, query: str):
        '''
        Main functionality of the engine. Takes a query as a parameter and returns
        a list of all URLs that are relevant to that query.
        '''
        # self.weighImportantTerms(5)
        # Parsing search query, getting the terms as stems
        # Tokens contains a dictionary of the token itself as the key and the position as the value
        tokens = self.parseSearch(query)

        # Dictionary to hold IDF information
        termIDFs = dict()

        postingList = []

        # Getting the appropriate information from the inverted index by seeking() to the correct position.
        for token in tokens:
            seek_position = int(self._indexOfIndex[token])
            self._invertedIndexFile.seek(seek_position)

            # Getting line
            information = self._invertedIndexFile.readline().strip()
            splitInformation = information.split(',')

            # Getting individual information
            idfScore = splitInformation[1]
            termIDFs[token] = float(idfScore)
            documentFrequency = splitInformation[2]
            termPostingList = splitInformation[3:]
            postingList.append(information)

        results, intersectionlength = self.intersect(postingList)
        max_length = 0
        max_value = None

        if intersectionlength < 10:
            for key, value in results.items():
                if len(value) > max_length:
                    max_length = len(value)
                    max_value = value
        else:
            for key, value in results.items():
                if len(value) == intersectionlength:
                    max_value = value
                    break

        queryTokens = dict()
        for token in tokens: #goes through tokens dictionary and makes a new dictionary with key = token value = len of position
            queryTokens[token] = len(tokens[token])

        # Dictionary to hold documents TFIDF information
        documentTFIDFs = dict()

        #calculate the cosine score using two functions one generic and weighImportant takes into account the important words
        cosineScore = self.fastCosineScore(queryTokens, termIDFs, results, documentTFIDFs)
        cosineScore = self.weighImportantTermsVersion1(cosineScore, queryTokens)
        # Weighting TF-IDF Scores
        # Dictionary that holds the total tfidf score for
        finalDocumentTFIDFScores = self.getDocumentTotalTFIDFScore(documentTFIDFs)

        # Combining all scores
        combinedScores = []
        for docID in cosineScore:
            combinedScores.append((-1 * (1 * cosineScore[docID] + 1 * finalDocumentTFIDFScores[docID]), docID))

        heapq.heapify(combinedScores)

        self.printURLs(combinedScores)

    def printURLs(self, scoreList: list):
        '''
        Takes a list of tuples and returns the URLs.
        '''
        numURLs = 0
        for tup in scoreList:
            if numURLs == 10:
                break
            url = self._urlMappings[str(tup[1])]
            if not any(ext in url for ext in [".txt", ".jpg", ".png", ".ff"]) and not url.endswith(
                    ".txt") and "txt" not in url:
                print(f"{numURLs + 1}: {url}")
                numURLs += 1

    def getDocumentTotalTFIDFScore(self, documentTFIDFs):
        documentTFIDFScores = dict()
        documentTFIDFNormalizeWeights = dict()
        for term, postingInfo in documentTFIDFs.items():
            for tup in postingInfo:
                docID = tup[0]
                tfidfScore = tup[1]

                if docID not in documentTFIDFScores:
                    documentTFIDFScores[docID] = tfidfScore
                else:
                    documentTFIDFScores[docID] += tfidfScore
                    # documentTFIDFNormalizeWeights[docID] += tfidfScore * tfidfScore

        # for documentID, score in documentTFIDFScores.items():
        # documentTFIDFScores[documentID] = score / math.sqrt(documentTFIDFNormalizeWeights[documentID])

        # Getting min and max to normalize
        max = -1000
        min = 1000
        for docID, score in documentTFIDFScores.items():
            if score > max:
                max = score
            if score < min:
                min = score

        for docID, score in documentTFIDFScores.items():
            documentTFIDFScores[docID] = (score - min) / (max - min)

        return documentTFIDFScores

    def fastCosineScore(self, tokens, termIDFs, results, documentTFIDFs):
        scores = [0] * 55392 #makes a list of 55392 elements whose values are all 0
        docIDs = set() 

        normalizedValue = self.getNormalizedQueryValue(tokens, termIDFs) #utilizes the termIDfs to get a normalized value 
        # print(normalizedValue)

        for term in tokens.keys(): #goes throguh tokens where each term is the token/word
            for posting in results[term]: #goes through the posting lists in results based on the term 
                posting = posting.split(':')
                docID = int(posting[0])
                docIDs.add(docID) #appends the docID to the docID set 
                freq = int(posting[1])
                freq = 1 + math.log10(freq)
                scores[docID] += freq * normalizedValue[term]

                # Appending TFIDF information to the document itself
                if term not in documentTFIDFs:
                    documentTFIDFs[term] = []
                    documentTFIDFs[term].append((docID, freq * termIDFs[term]))
                else:
                    documentTFIDFs[term].append((docID, freq * termIDFs[term]))

        cosineSimScore = dict()
        for docID in docIDs: #goes through the docID set
            docID_str = str(docID) #makes all docID's strings
            docLength = self._docLengths.get(docID_str)
            if docLength is not None: 
                docLength_value = docLength[1] #gets the document length 
                # Perform any necessary string manipulation on docLength_value
                try:
                    cosineSimScore[docID] = scores[docID] / float(docLength_value)
                except ValueError:
                    # Handle the case where the value cannot be converted to float
                    continue
            else:
                continue

        return cosineSimScore

    '''
    gets the basic cosineSimScore and utilizes the important words to increase the basic score 
    '''
    def weighImportantTermsVersion1(self, cosineSimScore, tokens):
     
        for term in tokens:
            if len(term) > 2: 
                if term in self._boldIndex:
                    positionB = int(self._boldIndex[term])
                    self._boldIndexFIle.seek(positionB)
                    informationB = self._boldIndexFIle.readline().strip().split(',')
                    for key in cosineSimScore:
                        if str(key) in informationB:
                            cosineSimScore[key] += cosineSimScore[key] * 0.25
     
            if len(term) > 2:
                if term in self._titleIndex:
                    positionT = int(self._titleIndex[term])
                    self._titleIndexFIle.seek(positionT)
                    informationT = self._titleIndexFIle.readline().strip().split(',')
                    for key in cosineSimScore:
                        if str(key) in informationT:
                            cosineSimScore[key] += cosineSimScore[key] * .85

        return cosineSimScore




    def calculateNorm(self, term, weight_of_words):
        norm_sum = 0

        for value in weight_of_words.values():
            squared_value = value ** 2
            norm_sum += squared_value

        term_value = weight_of_words.get(term, 0)  # Retrieve the value of the term from the dictionary

        if norm_sum > 0:
            result = term_value / math.sqrt(norm_sum)
        else:
            result = 0

        return result

    def getNormalizedQueryValue(self, tokens: dict, idfScores: dict):
        '''
        Takes a dict of tokens and idfscore dictionary as input. Returns the normalized
        weighting of the query.
        '''
        normalizedSum = 0
        tokenWeights = dict()
        for token, frequency in tokens.items():

            tfwt = 0
            if frequency > 0:
                tfwt = 1 + math.log10(frequency)

            weight = tfwt * idfScores[token]
            tokenWeights[token] = weight

            normalizedSum += weight * weight

        normalizedSum = math.sqrt(normalizedSum)

        for token, weight in tokenWeights.items():
            tokenWeights[token] = weight / normalizedSum

        return tokenWeights

    def merge(self, parsedQuery):
        '''
        Function that gets the posting lists of all the tokens in parsedQuery dictionary.
        Returns a list of all of the document Postings that have all words.
        '''

    def getUrlMappings(self, documentIDs):
        returnable = []
        for documentID in documentIDs:
            returnable.append(self._urlMappings[documentID])

        return returnable

    def getIndexPositions(self, tokens: list):
        '''
        Takes a list of tokens as input. Returns a dictionary with the token as key
        and the seek() position as the value.
        '''
        positions = dict()
        for token in tokens:
            positions[token] = self._indexOfIndex[token]

        return positions

    def parseSearch(self, query: str):
        '''
        Tokenizes + stems the search query and returns a dictionary of the words
        as the keys and its positions in the query as the value. Boolean retrieval model
        only uses the keys, while a phrase query will use the values for positions.
        '''
        returnable = dict()
        position = 0

        for token in re.findall(self._reg, query):
            token = self._stemmer.stem(token)
            if token not in returnable:
                returnable[token] = [position]
            else:
                returnable[token].append(position)

            position += 1

        return returnable


def run():
    # Open the mappings file, opening the index, parse into a dictionary, and put it into searchengine object
    with open('../FinalDocuments/mapping.txt', 'r') as mappingsFile, open('../FinalDocuments/invertedIndexFinalWithSkips.txt',
                                                                       'r') as invIndexFile, open(
            '../FinalDocuments/documentLengths.txt', 'r') as lengthOfDocs, open(
            '../Important_Words_Files/boldWords.txt') as boldWordsFile, open(
            '../Important_Words_Files/headerWords.txt') as headerWordsFile, open(
            '../Important_Words_Files/titleWords.txt') as titleWordsFile:

        # CAUTION - This line of code takes a while! Make sure to only perform this initialization ONCE
        engine = SearchEngine(PorterStemmer(), re.compile('[a-zA-Z0-9]+'), invIndexFile, mappingsFile,
                              '../FinalDocuments/index_Of_The_Index_Final.txt', lengthOfDocs, boldWordsFile,
                              '../Important_Words_Files/bold_words_Index_of_Index.txt', headerWordsFile,
                              '../Important_Words_Files/header_words_Index_of_Index.txt', titleWordsFile,
                              '../Important_Words_Files/title_words_Index_of_Index.txt')

        while True:
            # Prompt user for input - user can exit with ctrl C
            userQuery = input(">>> ")
            start_time = time.perf_counter()  # Timer begins counting the time
            print(engine.getUrls(userQuery))
            end_time = time.perf_counter()  # End the timer
            execution_time = end_time - start_time

            # Convert the execution time to minutes, seconds, and milliseconds
            minutes = int(execution_time // 60)
            seconds = int(execution_time % 60)
            milliseconds = int((execution_time - int(execution_time)) * 1000)

            # Display the execution time
            print(f"Execution time: {minutes} minutes {seconds} seconds {milliseconds} milliseconds")


if __name__ == '__main__':
    run()
