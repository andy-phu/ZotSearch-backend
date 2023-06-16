import math

def add_Skip_List_Jumps(filename):
    with open(filename, "r") as f:
        all_lines = []
        for line in f:
            results = []
            currLine = line.strip().split(',')
            results.append(currLine[0])
            results.append(currLine[1])
            results.append(currLine[2])

            lengthOfPosting = int(currLine[2])
            if lengthOfPosting < 4:
                all_lines.append(currLine)
                continue
            amount_of_skips_jumps = int(math.sqrt(lengthOfPosting))
            skip_jump = lengthOfPosting // amount_of_skips_jumps

            index = 3
            constIndex = 3
            skip_jump_count = amount_of_skips_jumps
            stopHere = 0

            for i, term in enumerate(currLine[3:], start=0):
                if stopHere < skip_jump:
                    if skip_jump_count >= amount_of_skips_jumps:
                        if constIndex + amount_of_skips_jumps < len(currLine):
                            x = currLine[constIndex + amount_of_skips_jumps]
                            y = x.split(":")[0]
                            z = currLine.index(x) - index
                            results.append(f"{term}:{y}:{z}")
                            skip_jump_count = 0
                            stopHere += 1
                            skip_jump_count += 1
                            constIndex += 1
                        else:
                            results.append(term)
                            skip_jump_count += 1
                            constIndex += 1
                    else:
                        results.append(term)
                        skip_jump_count += 1
                        constIndex += 1
                else:
                    results.append(term)
                    skip_jump_count += 1
                    constIndex += 1
            all_lines.append(results)

        return all_lines


def create_Skip_List(indexFile):
    skipList = add_Skip_List_Jumps(indexFile)

    # Writing to file
    with open('invertedIndexFinalWithSkips.txt', 'w') as f:        # Creates a new inverted index with skip jumps
        for line in skipList:
            formatted_line = ','.join(line)
            f.write(formatted_line + '\n')


if __name__ == '__main__':
    create_Skip_List('invertedIndex.txt') # Takes in the inverted index that was creating originally