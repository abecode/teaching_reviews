from collections import defaultdict
from dataclasses import dataclass, asdict
import json
import os
import time
import subprocess
from sklearn.model_selection import train_test_split
import spacy
#from spacy import displacy
from spacy.cli.train import train
from spacy.cli.evaluate import evaluate
from spacy.tokens import DocBin
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from . import util as tr


def extract_metrics(fname="metrics.json"):
    # Load the JSON file
    with open("metrics.json") as f:
        metrics = json.load(f)

    # Check the result
    import pprint
    pprint.pprint(metrics)
    return metrics

def flatten_metrics(metrics):
    """takes the metrics and groups together all precision, recall, and fscores"""
    # Build the 'metrics' dictionary
    output = {}

    # Per-type metrics
    per_type = metrics.get("spans_sc_per_type", {})
    for label, scores in per_type.items():
        output[label] = {
            "p": scores.get("p", 0.0),
            "r": scores.get("r", 0.0),
            "f": scores.get("f", 0.0)
        }

    # Add token-level metrics
    output["tokens"] = {
        "p": metrics.get("token_p", 0.0),
        "r": metrics.get("token_r", 0.0),
        "f": metrics.get("token_f", 0.0)
    }

    # Add overall span metrics under a separate key if you like
    output["spans"] = {
        "p": metrics.get("spans_sc_p", 0.0),
        "r": metrics.get("spans_sc_r", 0.0),
        "f": metrics.get("spans_sc_f", 0.0)
    }

    # Check the result
    # import pprint
    # pprint.pprint(output)
    return output


def experiment_001_basic_spacy_word_vectors():
    """This is an experiment with just the basic spacy word vectors and the
    organic data split 70/15/15 using spans that were agreed upon by 3+
    annotators

    Note: this function uses half subprocess to run shell commands and half
    spacy spacy.cli . Subprocess wasn't working (maybe because of tqdm type status
    messages?)
    """

    prodigy_review_file = \
    "teaching_reviews/data_jsonl/teaching_reviews_pilot1_spans_reviews_20250422.json"

    # check that the data file exists
    if not os.path.exists(prodigy_review_file):
        raise Exception(f"data file {prodigy_review_file} not found")
    prodigy_df = tr.file_to_span_df(prodigy_review_file)
    prodigy_df = tr.filter_spans_with_gte3_agreement(prodigy_df)
    # Group by input_hash to collect spans for the same document
    # this way there will be no spans from the same document crossing
    # train/dev/test partitions
    grouped = defaultdict(list)
    for row in prodigy_df.itertuples(index=False):
        grouped[row.input_hash].append(row)

    # Load a blank or pretrained spaCy pipeline
    nlp = spacy.blank("en")
    # or spacy.load("en_core_web_sm") if you're using a tokenizer

    # Get input_hash keys
    input_hashes = list(grouped.keys())

    # Use scikit-learn for stratified splitting (or simple random)
    train_ids, temp_ids = train_test_split(input_hashes, test_size=0.3,
                                           random_state=42)
    dev_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    split_ids = {
        "train": train_ids,
        "dev": dev_ids,
        "test": test_ids,
    }

    # Split input_hash keys so that we don't have any of the same documents
    # across train/dev/test sets
    input_hashes = list(grouped.keys())
    # Use scikit-learn for stratified splitting (or simple random)
    train_ids, temp_ids = train_test_split(input_hashes,
                                           test_size=0.3,
                                           random_state=42) # 70% training
    dev_ids, test_ids = train_test_split(temp_ids, test_size=0.5,
                                         random_state=42)

    nlp = spacy.blank("en")

    split_doc_bins = {
        "train": DocBin(store_user_data=True),
        "dev": DocBin(store_user_data=True),
        "test": DocBin(store_user_data=True),
    }

    #add the
    for split_name, hash_list in split_ids.items():

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
            split_doc_bins[split_name].add(doc)

    print("train/dev/test document counts", len(split_doc_bins['train']),
          len(split_doc_bins['dev']), len(split_doc_bins['test']))

    # save data to file
    for split_name, doc_bin in split_doc_bins.items():
        output_file = f"{split_name}.spacy"
        doc_bin.to_disk(output_file)
        print(f"✅ Saved {split_name} data to {output_file}")

    # generate spacy config:
    # python -m spacy init config config.cfg --pipeline spancat --lang en --force
    try:
        result = subprocess.run(["python", "-m", "spacy", "init", "config",
                                 "config.cfg", "--pipeline", "spancat",
                                 "--lang", "en", "--force"],
                                capture_output=True, text=True)
        print(result.stdout)
    except:
        raise Exception("⚠️ spacy init config failed")

    # set space spancat suggester config to have ngram size up to 30
    # perl -pe  's/sizes = \[1,2,3\]/sizes = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30\]/' -i.bak config.cfg
    # cat config.cfg | grep sizes
    try:
        result = subprocess.run(["perl", "-pe", "s/sizes = \[1,2,3\]/sizes = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30\]/",
                                 "-i.bak", "config.cfg"],
                                capture_output=True, text=True)
        print(result.stdout)
        # cat config.cfg | grep sizes
        result = subprocess.run(["cat", "config.cfg", "|", "grep", "sizes"],
                                capture_output=True, text=True)
        print(result.stdout)
        if "30" not in result.stdout:
            raise Exception("⚠️ ngram size not set to 30")
    except:
        raise Exception("⚠️ spacy init config failed")

    # in case of trouble try this
    # python -m spacy debug data config.cfg  --paths.train ./train.spacy  --paths.dev ./dev.spacy --paths.test ./test.spacy
    # result = subprocess.run("python -m spacy debug data config.cfg  --paths.train ./train.spacy  --paths.dev ./dev.spacy --paths.test ./test.spacy",
    #                         shell=True, capture_output=True, text=True)
    # print("debug config results")
    # print(result.stdout)

    # train
    #train_cmd = "python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy --gpu-id 0"
    # training was hard to get working with subprocess
    # https://github.com/explosion/spaCy/discussions/11673
    start = time.time()
    train("./config.cfg", use_gpu=0, output_path="output",
          overrides={"paths.train": "./train.spacy", "paths.dev": "./dev.spacy"})
    end = time.time()
    print(f"Training took {end - start:.2f} seconds.")

    # evaluate
    result = evaluate("output/model-best", "./test.spacy", "metrics.json", use_gpu=0)

    #
    print("eval results")
    print(result)

    return extract_metrics()

def experiment_002_basic_spacy_word_vectors_plus_gpt_data():
    """This is an experiment with  spacy word vectors and the
    organic data split 70/15/15 using spans that were agreed upon by 3+
    annotators, with additional gpt data added to the test set

    Note: this function uses half subprocess to run shell commands and half
    spacy spacy.cli . Subprocess wasn't working (maybe because of tqdm type status
    messages?)
    """

    prodigy_review_file = "teaching_reviews/data_jsonl/teaching_reviews_pilot1_spans_reviews_20250422.json"

    # check that the data file exists
    if not os.path.exists(prodigy_review_file):
        raise Exception(f"data file {prodigy_review_file} not found")
    prodigy_df = tr.file_to_span_df(prodigy_review_file)
    prodigy_df = tr.filter_spans_with_gte3_agreement(prodigy_df)
    # Group by input_hash to collect spans for the same document
    # this way there will be no spans from the same document crossing
    # train/dev/test partitions
    grouped = defaultdict(list)
    for row in prodigy_df.itertuples(index=False):
        grouped[row.input_hash].append(row)

    # Load a blank or pretrained spaCy pipeline
    nlp = spacy.blank("en")  # or spacy.load("en_core_web_sm") if you're using a tokenizer

    # Get input_hash keys
    input_hashes = list(grouped.keys())

    # Use scikit-learn for stratified splitting (or simple random)
    train_ids, temp_ids = train_test_split(input_hashes, test_size=0.3, random_state=42)
    dev_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    split_ids = {
        "train": train_ids,
        "dev": dev_ids,
        "test": test_ids,
    }

    # Split input_hash keys so that we don't have any of the same documents
    # across train/dev/test sets
    input_hashes = list(grouped.keys())
    # Use scikit-learn for stratified splitting (or simple random)
    train_ids, temp_ids = train_test_split(input_hashes, test_size=0.3, random_state=42) # 70% training
    dev_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    nlp = spacy.blank("en")

    ####### Load gpt data
    gpt_spans = []
    linenum = 0
    for line in open("teaching_reviews/data_chatgpt/annotated_reviews_manual.txt"):
        line = line.strip()
        if not line:
            continue

        spacy_fmt = tr.bracket_to_spacy(line)
        spacy_fmt['linenum'] = linenum
        linenum += 1
        gpt_spans.append(spacy_fmt)

    def spans_to_df(span_dict_list):
        spans = []
        for span_dict in span_dict_list:
            for span in span_dict['spans']:
                spans.append(tr.Span(span["start"], span["end"], span["label"], span_dict["input_hash"], span_dict["filename"], span_dict["linenum"], span_dict["annotator"], span_dict["text"][span["start"]:span["end"]], span_dict["text"]))
        return pd.DataFrame([asdict(span) for span in spans])

    gpt_df = spans_to_df(gpt_spans)

    split_doc_bins = {
        "train": DocBin(store_user_data=True),
        "dev": DocBin(store_user_data=True),
        "test": DocBin(store_user_data=True),
    }

    for split_name, hash_list in split_ids.items():

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
            split_doc_bins[split_name].add(doc)

    print("train/dev/test document counts before adding chatgpt data", len(split_doc_bins['train']),
          len(split_doc_bins['dev']), len(split_doc_bins['test']))


    # Group by input_hash to collect spans for the same document
    grouped = defaultdict(list)
    for row in gpt_df.itertuples(index=False):
        grouped[row.input_hash].append(row)

    # Load a blank or pretrained spaCy pipeline
    nlp = spacy.blank("en")  # or spacy.load("en_core_web_sm") if you're using a tokenizer

    # Get input_hash keys
    input_hashes = list(grouped.keys())


    #add the rows to the training set
    for input_hash in input_hashes:
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
                print(f"⚠️ Skipping invalid span in gpt data: "
                      "{ex.label} {ex.start}-{ex.end}")

        doc.spans["sc"] = span_objs
        split_doc_bins["train"].add(doc)


    print("train/dev/test document counts after adding chatgpt data", len(split_doc_bins['train']),
          len(split_doc_bins['dev']), len(split_doc_bins['test']))

    # save data to file
    for split_name, doc_bin in split_doc_bins.items():
        output_file = f"{split_name}.spacy"
        doc_bin.to_disk(output_file)
        print(f"✅ Saved {split_name} data to {output_file}")


    # generate spacy config:
    # python -m spacy init config config.cfg --pipeline spancat --lang en --force
    try:
        result = subprocess.run(["python", "-m", "spacy", "init", "config",
                                 "config.cfg", "--pipeline", "spancat",
                                 "--lang", "en", "--force"],
                                capture_output=True, text=True)
        print(result.stdout)
    except:
        raise Exception("⚠️ spacy init config failed")

    # set space spancat suggester config to have ngram size up to 30
    # perl -pe  's/sizes = \[1,2,3\]/sizes = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30\]/' -i.bak config.cfg
    # cat config.cfg | grep sizes
    try:
        result = subprocess.run(["perl", "-pe", "s/sizes = \[1,2,3\]/sizes = \[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30\]/",
                                 "-i.bak", "config.cfg"],
                                capture_output=True, text=True)
        print(result.stdout)
        # cat config.cfg | grep sizes
        result = subprocess.run(["cat", "config.cfg", "|", "grep", "sizes"],
                                capture_output=True, text=True)
        print(result.stdout)
        if "30" not in result.stdout:
            raise Exception("⚠️ ngram size not set to 30")
    except:
        raise Exception("⚠️ spacy init config failed")

    # in case of trouble try this
    # python -m spacy debug data config.cfg  --paths.train ./train.spacy  --paths.dev ./dev.spacy --paths.test ./test.spacy
    # result = subprocess.run("python -m spacy debug data config.cfg  --paths.train ./train.spacy  --paths.dev ./dev.spacy --paths.test ./test.spacy",
    #                         shell=True, capture_output=True, text=True)
    # print("debug config results")
    # print(result.stdout)

    # train
    #train_cmd = "python -m spacy train config.cfg --output ./output --paths.train ./train.spacy --paths.dev ./dev.spacy --gpu-id 0"
    # training was hard to get working with subprocess
    # https://github.com/explosion/spaCy/discussions/11673
    start = time.time()
    train("./config.cfg", use_gpu=0, output_path="output",
          overrides={"paths.train": "./train.spacy", "paths.dev": "./dev.spacy"})
    end = time.time()
    print(f"Training took {end - start:.2f} seconds.")
    evaluate("output/model-best", "./test.spacy", "metrics.json", use_gpu=0)

    return extract_metrics()



def plot_metrics(metrics):
    """ plots Data from your metrics.json after it has been flattened to
    transform all the p/r/f scores into the same table"""

    labels = set(metrics.keys())
    precision = [metrics[label]["p"] for label in labels]
    recall = [metrics[label]["r"] for label in labels]
    f1 = [metrics[label]["f"] for label in labels]

    x = range(len(labels))

    plt.figure(figsize=(12,6))
    plt.bar(x, precision, width=0.2, label="Precision", align='center')
    plt.bar([i + 0.2 for i in x], recall, width=0.2, label="Recall", align='center')
    plt.bar([i + 0.4 for i in x], f1, width=0.2, label="F1", align='center')

    plt.xticks([i + 0.2 for i in x], labels, rotation=45)
    plt.ylim(0,1)
    plt.ylabel("Score")
    plt.title("Span Classification Metrics by Label")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,6))
    plt.scatter(precision, recall, s=100)

    # get rid of the token and span aggregate p/r/f
    # labels = set(metrics.keys()) - set(["tokens", "spans"])
    for i, label in enumerate(labels):
        plt.annotate(label, (precision[i]+0.01, recall[i]+0.01))

    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.title("Precision vs Recall by Label")
    plt.grid(True)
    plt.show()

def plot_metrics_heatmap(metrics):
    """ plots a heatmap of the metrics by label"""
    df = pd.DataFrame(metrics).T
    sns.heatmap(df, annot=True, cmap="YlGnBu", vmin=0, vmax=1)
    plt.title("Per-Label Metrics Heatmap")
    plt.show()

def make_experiment_results_table(experiment_results):
    """ takes the experiemental results and outputs a pandas table with
    columns for each metric and rows for each experiment"""
    output = pd.DataFrame(columns=["experiment", "metric", "precision", "recall", "f1"])
    for experiment_name in experiment_results:
        nested_rows = flatten_metrics(experiment_results[experiment_name])
        for metric, row in nested_rows.items():
            output.loc[len(output)] = [experiment_name, metric, row["p"], row["r"], row["f"]]
    return output