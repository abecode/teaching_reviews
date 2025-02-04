#!/usr/bin/env python
# doh, something happened where python3 points to homebrew
# damn you, homebrew!
"""
split the sentences from a directory
"""
import os
import sys
import typing

import spacy
nlp = spacy.load("en_core_web_sm")

directory = "data"

if len(sys.argv) > 1:
    directory = sys.argv[1]

#print(directory)

# Construction via add_pipe
sentencizer = nlp.add_pipe("sentencizer")

# Construction from class
#from spacy.pipeline import Sentencizer
#sentencizer = Sentencizer()

# for fname in os.listdir(directory):
#     with open(os.path.join(directory, fname)) as f:
#         for line in f:
#             #print(line.strip())
#             doc = nlp(line)
#             for sent in doc.sents:
#                 print(fname, sep="\t")
#                 print(sent)

def split_sentences(f_in: typing.TextIO) -> typing.Generator:
    for line in f_in:
        doc = nlp(line)
        for sent in doc.sents:
            yield sent


if __name__ == "__main__":
    # piping in, just read in and print out
    if not sys.stdin.isatty(): 
        for x in split_sentences(sys.stdin):
            print(x)
    # batch usage, process a whole directory
    else:
        try:
            rawdatadir = os.environ["RAWDATADIR"]
            splitdatadir = os.environ["SPLITDATADIR"]
            for _file in os.listdir(rawdatadir):
                if _file.endswith(".txt"):
                    with open(os.path.join(rawdatadir, _file)) as F_in, open(os.path.join(splitdatadir, _file), "w") as F_out:
                        for line in split_sentences(F_in):
                            print(line, file=F_out)
        except KeyError:
            print("to run this set RAWDATADIR and SPLITDATADIR", file=sys.stderr)
            print("e.g.: ", file=sys.stderr)
            print("  export RAWDATADIR=data", file=sys.stderr)
            print("  export SPLITDATADIR=split_data", file=sys.stderr)
            print("  or RAWDATADIR=data SPLITDATADIR=split_data ./split_sentences.py",
                  file=sys.stderr)
            print("It is recommended to have data backed up and/or in version control",
                  file=sys.stderr)            
            

