#!/usr/bin/env python3
"""previous script, ./extract_reviews_by_sentence_to_json.py, used
spacy to do sentence segmentation, which had some errors, e.g.

201940SEIS631-01.txt
I give this professor a lot of credit for choosing a solid text, responding quickly to questions via a variety of forms of communication, and for soliciting and acting on feedback to improve his class.
He's approachable, knowledgeable and I believe he cares about the success of his students.
Unfortunately, like so many classes in the grad software program, this class is characterized by: 1.
An assignment to read a chapter 2.
A class session spent going over PowerPoint slides that reiterate what was in the chapter but without bringing focus or clarity to the material 3.
A smallish assignment that checks the box for "hands on work" 4.
Two massive tests that comprise 60-80% of the grade (60% in this particular class) I believe this class, like so many others in the grad software program, would benefit from being flipped.

so instead of converting directly to sentences with spacy, I saved the
split data to /split_data, checked it into git, and then manually
fixed these sentence segmentation issues


so this script will convert the data from its existing format (directory of files, one
sentence per line, separate reviews separated by a blank line) to jsonl format
{"text": <review>
 "meta": {"filename": <fname>,
          "linenum": <linenum> # starting at 0
         }
}

"""
import os
import sys
import json

directory = "split_data"

if len(sys.argv) > 1:
    directory = sys.argv[1]

    
for fname in os.listdir(directory):
    with open(os.path.join(directory, fname)) as f:
        linenum = 0   # this is the original line number, ie the review
        sentnum = 0   # this is the sentence number after spacy split and manual correction
        for line in f:
            if not line or line.isspace():
                linenum += 1
                sentnum = 0
                continue
            outdict = {"text": line.strip(),
                       "meta": {"filename": fname,
                                "linenum": linenum,
                                "sentnum": sentnum,
                                }
                       }
            sentnum += 1
            print(json.dumps(outdict))


