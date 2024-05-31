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

directory = "data"

if len(sys.argv) > 1:
    directory = sys.argv[1]

    
for fname in os.listdir(directory):
    with open(os.path.join(directory, fname)) as f:
        linenum = 0
        for line in f:
            outdict = {"text": line.strip(),
                       "meta": {"filename": fname,
                                "linenum": linenum
                                }
                       }
            print(json.dumps(outdict))
            linenum += 1

