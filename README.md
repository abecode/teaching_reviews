# Teaching Reviews Data and Analysis

## First step: extract sentences from data

```
pip install spacy
python -m spacy download en_core_web_sm
python split_sentences.py
```

## Sentiment Analysis Example Using Textblob

```
pip install spacytextblob
python -m textblob.download_corpora
# this failed, but I found
# https://stackoverflow.com/questions/41691327/ssl-sslerror-ssl-certificate-verify-failed-certificate-verify-failed-ssl-c/41692664#41692664
# and added the lines
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# to
site-packages/textblob/download_corpora.py
# and it worked (I haven't had time to analyze the actual results)
python textblob_sentiment.py

```


## Annotation

### whole review

prodigy command to annotate whole reviews:

```
export PRODIGY_BASIC_AUTH_USER=.......
export PRODIGY_BASIC_AUTH_PASS=.......
prodigy textcat.manual teaching_reviews_whole_review ./teaching_reviews_whole_review.jsonl --label POSITIVE,NEGATIVE,NEUTRAL,MIXED,UNSURE
PRODIGY_HOST=0.0.0.0 prodigy textcat.manual teaching_reviews_whole_review ./data_jsonl/teaching_reviews_whole_review.jsonl --label POSITIVE,NEGATIVE,NEUTRAL,MIXED,UNSURE
```

to export mengyuan's annotations:

```
prodigy db-out teaching_reviews_whole_review > mengyuan_teaching_reviews_whole_review.jsonl
# copy from server: ssh -t ec2-user@server 'cat prodigy_teaching_reviews/mengyuan_teaching_reviews_whole_review.jsonl' > teaching_reviews_whole_review_POSITIVE,NEGATIVE,NEUTRAL,MIXED,UNSURE_mengyuan_20240327.jsonl
```

### sentence level annotations


prodigy.json settings

```
{
"feed_overlap":true
}
```


[Instructions for sentence annotations](docs/sentence_annotation.md)

Prodigy command to annotate reviews split by sentence:

```
export PRODIGY_BASIC_AUTH_USER=.......
export PRODIGY_BASIC_AUTH_PASS=.......
export PRODIGY_ALLOWED_SESSIONS=...... # names of annotators
prodigy textcat.manual teaching_reviews_sentences ./data_jsonl/teaching_reviews_sentences.jsonl --label POSITIVE,NEGATIVE,NEUTRAL,MIXED,UNSURE,SUGGESTION
```


### Span Annotations

c.f. [SpaCy docs](https://prodi.gy/docs/span-categorization)

first, try sentence-level span annotations 

```
prodigy spans.manual tmp_spans blank:en ./data_jsonl/teaching_reviews_sentences.jsonl --label POSITIVE,NEGATIVE,NEUTRAL,UNSURE,SUGGESTION
```

realize: there can still be mixed sentiment at the span level: comparisons are neither positive nor negative but their own label

Also, it doesn't make sense to label spans at the sentence level.  Sentence splitting has some issues and annotating a whole

```
prodigy spans.manual tmp2_spans blank:en ./data_jsonl/teaching_reviews_whole_review.jsonl --label POSITIVE,NEGATIVE,SUGGESTION,COMPARISON,UNSURE,REDACT
# remove neutral
```

save data to data_prodigy/prodigy.20250325.db
