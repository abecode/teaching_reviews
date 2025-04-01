#!/usr/bin/env python3

""" expects a pipe containing prodigy style json annotation data
e.g. 
prodigy db-out pilot1_spans | ./analyze_annotations.py
or cat file.json | ./analyze_annotations.py

"""

import json
from collections import defaultdict
import sys

annotator_stats = defaultdict(int)


# anntations = sys.argv[1]

# with open(sys.stdin, 'r', encoding='utf-8') as f:
#     for line in f:
#         rec = json.loads(line)
#         annotator = rec.get('_annotator_id', 'unknown')
#         annotator_stats[annotator] += 1

for line in sys.stdin:
    rec = json.loads(line)
    annotator = rec.get('_annotator_id', 'unknown');
    annotator_stats[annotator] += 1

for annotator, count in annotator_stats.items():
    print(f'Annotator: {annotator}, Annotations: {count}')
