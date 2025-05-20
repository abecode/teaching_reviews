""" module docstring"""


from collections import defaultdict #, Counter
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import List, Type
from IPython.display import display, HTML
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import spacy
from spacy import displacy
from spacy.tokens import DocBin

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

def extract_spans_from_review(span_list: dict) -> List[tuple]:
    """Convert spans to (start, end, label) format. just for a single review"""
    return [(s["start"], s["end"], s["label"]) for s in span_list]

def extract_spans(review_list: List[tuple]) -> List[Type[pd.DataFrame]]:
    """take a list of loaded json objects and output a dataframe
    who's rows are derived from Span objects"""
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
    return pd.DataFrame([asdict(span) for span in spans])

def file_to_span_df(path: str) -> pd.DataFrame:
    """ go directly from the json file to a dataframe"""
    return extract_spans(load_jsonl(path))

def filter_spans_with_gte3_agreement(all_spans: pd.DataFrame) -> pd.DataFrame:
    """ extract spans with 3 or more agreeing annotators """
    all_spans["count"] = np.ones(len(all_spans))
    spans_with_counts = all_spans.groupby(["input_hash", "start", "end", "label",
                                           "filename", "linenum", "span"])\
                                 .count()\
                                 .sort_values(["filename", "linenum"])\
                                 .drop(columns=["annotator", "text"])\
                                 .reset_index()
    gte3 = spans_with_counts[spans_with_counts["count"] >=3 ]
    return gte3

def write_spacy_train_dev_test(df: pd.DataFrame, sanity_check=False) -> None:
    """ writes spacy train, dev, and test data files """
    # pylint: disable=too-many-locals
    # Group by input_hash to collect spans for the same document
    grouped = defaultdict(list)
    for row in df.itertuples(index=False):
        grouped[row.input_hash].append(row)

    # Load a blank or pretrained spaCy pipeline
    nlp = spacy.blank("en")  # or spacy.load("en_core_web_sm") if you're using a tokenizer

    # Split input_hash keys
    input_hashes = list(grouped.keys())

    # Use scikit-learn for stratified splitting (or simple random)
    train_ids, temp_ids = train_test_split(input_hashes, test_size=0.3, random_state=42)
    dev_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    splits = {
      "train": train_ids,
      "dev": dev_ids,
      "test": test_ids,
    }


    # Create and write .spacy files
    nlp = spacy.blank("en")

    for split_name, hash_list in splits.items():
        doc_bin = DocBin(store_user_data=True)

        for input_hash in hash_list:
            examples = grouped[input_hash]
            text = examples[0].text
            doc = nlp.make_doc(text)

            span_objs = []
            for ex in examples:
                span = doc.char_span(ex.start, ex.end, label=ex.label,
                                     alignment_mode="contract")
                if span:
                    span_objs.append(span)
                else:
                    print(f"⚠️ Skipping invalid span in {split_name}: "
                          "{ex.label} {ex.start}-{ex.end}")

            doc.spans["sc"] = span_objs
            doc_bin.add(doc)

        output_file = f"{split_name}.spacy"
        doc_bin.to_disk(output_file)
        print(f"✅ Saved {split_name} data to {output_file}")

    if sanity_check:  #sanity check: dump jsonl for each split
        for split_name, hash_list in splits.items():
            output_path = Path(f"{split_name}_debug.jsonl")
            with output_path.open("w", encoding="utf-8") as out:
                for h in hash_list:
                    examples = grouped[h]
                    text = examples[0].text
                    spans = [{"start": ex.start, "end": ex.end, "label": ex.label}
                             for ex in examples]
                    record = {
                        "text": text,
                        "spans": spans,
                        #"annotator": examples[0].annotator,  # Optional
                        "input_hash": h
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"✅ Wrote {split_name}_debug.jsonl for inspection")

        # load SpaCy model (for tokenizer)
        nlp = spacy.blank("en")

        # choose your split
        jsonl_path = Path("train_debug.jsonl")  # or dev_debug.jsonl / test_debug.jsonl

        # load JSONL and parse into spaCy Docs
        examples = []
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                doc = nlp.make_doc(record["text"])
                spans = []
                for span in record["spans"]:
                    s = doc.char_span(span["start"], span["end"], label=span["label"],
                                      alignment_mode="contract")
                    if s is not None:
                        spans.append(s)
                    else:
                        print(f"⚠️ Skipping invalid span: {span}")
                doc.spans["sc"] = spans  # This mimics spancat training
                examples.append(doc)

        # display first N examples
        for doc in examples[:5]:
            display(HTML(displacy.render(doc, style="span", options={"spans_key": "sc"})))
