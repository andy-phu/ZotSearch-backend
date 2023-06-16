





if __name__ == '__main__':
    bold_words_index_of_index = dict()
    title_words_index_of_index = dict()
    header_words_index_of_index = dict()

    with open('bold_words_Index_of_Index.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            key = parts[0]
            value = int(parts[1])
            bold_words_index_of_index[key] = value

    with open('title_words_Index_of_Index.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            key = parts[0]
            value = int(parts[1])
            title_words_index_of_index[key] = value

    with open('header_words_Index_of_Index.txt', 'r') as f:
        for line in f:
            parts = line.strip().split()
            key = parts[0]
            value = int(parts[1])
            header_words_index_of_index[key] = value


    boldWord = "bryan"
    listBold = []
    with open('boldWords.txt', "r") as file:
        positionOfBold = bold_words_index_of_index[boldWord]
        file.seek(positionOfBold)
        line = file.readline().strip()
        listBold.append(line)
    print(listBold)


    titleWord = "klefstad"
    listTitle = []
    with open('titleWords.txt', "r") as file:
        positionOfTitle = title_words_index_of_index[titleWord]
        file.seek(positionOfTitle)
        line = file.readline().strip()
        listTitle.append(line)
    print(listTitle)


    headerWord = "bryan"
    listHeader = []
    with open('headerWords.txt', "r") as file:
        positionOfHeader = header_words_index_of_index[headerWord]
        file.seek(positionOfHeader)
        line = file.readline().strip()
        listHeader.append(line)
    print(listHeader)