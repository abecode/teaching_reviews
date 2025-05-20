""" module docstring"""


#from collections import defaultdict, Counter
from dataclasses import dataclass#, asdict
import json
#from pathlib import Path
from typing import List, Type

@dataclass
class Span:
    """ all the info we want from a span in a dataclass """
    # pylint: disable=too-many-instance-attributes
    start: int
    end: int
    label: str
    input_hash: str
    filename: str
    linenum: int
    annotator: str
    span: str
    text: str

def load_jsonl(path:str) -> List[dict]:
    """ loads a json file from the path"""
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def extract_spans_from_review(span_list:dict) -> List[tuple]:
    """Convert spans to (start, end, label) format. just for a single review"""
    return [(s["start"], s["end"], s["label"]) for s in span_list]

def extract_spans(review_list:List[tuple]) -> List[Type[Span]]:
    """ take a list of loaded json objects and output a list of Span objects """
    spans = []
    for eg in review_list:
        if "spans" in eg and "answer" in eg and eg["answer"] == "accept":
            reviewer = eg.get("_reviewer_id") or eg.get("_session_id") or "unknown"
            filename = eg['meta']["filename"]
            linenum = eg['meta']["linenum"]
            input_hash = eg["_input_hash"]
            for start, end, label in extract_spans_from_review(eg["spans"]):
                spans.append(Span(start, end, label, input_hash, filename,
                             linenum, reviewer, eg["text"][start:end],
                             eg["text"]))
    return spans
