def get_schemes(text, schemes_citation):
    '''
    Function to return the numbers of enron schemes citation on a text

    :param text: list with the words on the text

    :return schemes_citation: number of citation of enron schemes on the text
    '''
    california_schemes = ["fat boy", "inc-ing", "ricochet", "megawatt laundering", "black window", "big foot",
                          "get shorty", "death star", "cong catcher", "perpetual loop", "red congo"]

    for pos in range(len(text)):
        # For one-word schemes just look on the list
        if text[pos] in california_schemes:
            schemes_citation += 1

        # For two-word schemes, it is necessary to check two words
        elif ((text[pos - 1] == 'fat') and (text[pos] == 'boy')):
            schemes_citation += 1
        elif ((text[pos - 1] == 'megawatt') and (text[pos] == 'laundering')):
            schemes_citation += 1
        elif ((text[pos - 1] == 'black') and (text[pos] == 'window')):
            schemes_citation += 1
        elif ((text[pos - 1] == 'big') and (text[pos] == 'foot')):
            schemes_citation += 1
        elif ((text[pos - 1] == 'get') and (text[pos] == 'shorty')):
            schemes_citation += 1
        elif ((text[pos - 1] == 'death') and (text[pos] == 'star')):
            schemes_citation += 1
        elif ((text[pos - 1] == 'cong') and (text[pos] == 'catcher')):
            schemes_citation += 1
        elif ((text[pos - 1] == 'perpetual') and (text[pos] == 'loop')):
            schemes_citation += 1
        elif ((text[pos - 1] == 'red') and (text[pos] == 'congo')):
            schemes_citation += 1

    return schemes_citation
