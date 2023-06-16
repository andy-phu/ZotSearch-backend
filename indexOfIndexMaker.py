# Parses the inverted index and writes an index of that index to a file.

# Expected Inverted Index Format: [token] [frequency] [documentID:frequency:position] [documentID:frequency:position] ...
# Writing to file format: [token] [position to seek()]


def convert(filename: str):
    with open(filename, "r") as f:
        offset = 0
        returnable = dict()

        for sline in f:
            line = sline.split(',')

            # Attributes
            token = line[0]

            returnable[token] = offset

            # Adding string line length to offset
            offset += len(sline)

        return returnable


def getIndexOfIndex(invertedIndexFile: str):
    indexOfIndex = convert(invertedIndexFile)

    # Writing to file
    with open('index_Of_The_Index_Final.txt', 'w') as f:
        for key, value in indexOfIndex.items():
            f.write(f"{key} {value}\n")


if __name__ == '__main__':
    getIndexOfIndex('invertedIndexFinalWithSkips.txt')