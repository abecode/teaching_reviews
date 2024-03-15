#!/usr/bin/env python3
"""
get basic stats for the data
"""
import os
import sys

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

class_count = 0
review_count = 0
sentence_count = 0

for fname in os.listdir(directory):
    class_count += 1
    with open(os.path.join(directory, fname)) as f:
        for line in f:
            review_count += 1
            #print(line.strip())
            doc = nlp(line)
            for sent in doc.sents:
                sentence_count += 1

print(f"class count = {class_count}")
print(f"review_count = {review_count}")
print(f"sentence_count = {sentence_count}")
