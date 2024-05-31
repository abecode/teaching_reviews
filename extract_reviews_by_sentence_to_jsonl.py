#!/usr/bin/env python3
"""convert the data from its existing format (directory of files, one
review per line) to jsonl format
{"text": <review>
 "meta": {"filename": <fname>,
          "linenum": <linenum> # starting at 0
         }
}

"""
import os
import sys
import json

import spacy
nlp = spacy.load("en_core_web_sm")

directory = "data"

if len(sys.argv) > 1:
    directory = sys.argv[1]

    
for fname in os.listdir(directory):
    with open(os.path.join(directory, fname)) as f:
        linenum = 0
        for line in f:
            sentnum = 0
            doc = nlp(line)
            for sent in doc.sents:
                outdict = {"text": sent.text.strip(),
                           "meta": {"filename": fname,
                                    "linenum": linenum,
                                    "sentnum": sentnum,
                                    }
                           }
                sentnum += 1
                print(json.dumps(outdict))
            linenum += 1

