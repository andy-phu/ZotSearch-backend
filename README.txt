README.txt  file describing how to use your software (i.e. at least: how to run the code that creates the index, how to start the search interface and how to perform a simple query).


Index Creation:
To create an index, we developed a function that processes a collection of JSON files located in the DEV folder. The text is tokenized, and an inverted index is generated, including word frequencies and page IDs. Subsequently, the index and URL mappings are saved in separate files. The index contains the word frequencies and page IDs, while the mappings store the associations between page IDs and their respective URLs.


We utilized several important libraries for this task. The nltk library was employed for implementing the PorterStemmer, while bs4 facilitated HTML tag handling through BeautifulSoup. Additionally, we utilized the json library for reading and parsing JSON files, and the urllib.parse library for urldefrag, which helped us extract the meaningful part of the URLs.


Search Interface:
To initiate the search interface, execute the engine.py file with the appropriate text files that require processing. The engine.py file processes the invertedIndexFinalWithSkips.txt, which enables efficient processing, and the documentLengths.txt, which provides a posting of URLs along with their corresponding lengths. Furthermore, the boldWords.txt, headerWords.txt, and titleWords.txt files are processed, as they are utilized to assign extra score values to words appearing in bold, headers, or titles. Titles receive the highest weight, followed by headers, and finally, bolded text. The respective indexes of all the mentioned files are also required.


Once the necessary files have been read and processed, the code prompts the user to enter their query, with no limitation on the number of words. After the user submits the query, a list of relevant URLs is displayed based on our team's ranking algorithm. Our algorithm employs cosine similarity to assign ranking scores to documents. The concise similarity function utilizes the library numpy to calculate the dot product which is essential in calculating the cosine similarity. Once the results are displayed, users have the option to enter different queries if desired.