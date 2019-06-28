#!/usr/bin/python

import string

def parseOutText(f):
    """
        Code original from Udacity Class

        given an opened email file f, parse out all text below the
        metadata block at the top and return a list that contains all the words in the email

        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(str.maketrans("", "", string.punctuation))

        ### split the text string into individual words
        words = text_string.split()

    return words

