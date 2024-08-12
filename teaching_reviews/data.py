import json
import os
import pandas as pd
import sqlite3
labels = ["UNSURE", "MIXED", "POSITIVE", "NEGATIVE", "NEUTRAL", "SUGGESTION"]

def get_current_annotation_db_connection() -> sqlite3.Connection:
    """ opens the current database file """
    dbfile = "prodigy.sentence-level.20240701.db"
    dbpath =  os.path.join(os.path.dirname(__file__),
                           dbfile)
    return sqlite3.connect(dbpath)

def get_sentence_annotation_pivot_df(conn: sqlite3.Connection) -> pd.DataFrame:
    """ gets a pivot table of the annotations

    ie a table/dataframe where each row is a sentence of the input and each column
    is an annotator's response. There are also columns for the text that was annotated
    and the Prodigy task_hash """

    # query result column indexes
    dataset_id_idx = 0
    dataset_name_idx = 1
    dataset_created_idx = 2
    dataset_meta_idx = 3
    dataset_session_idx = 4
    example_id_idx = 5
    example_input_hash_idx = 6
    example_task_hash_idx = 7
    example_content_idx = 8

    abe_annotations = conn.execute(
        """select dataset.id, dataset.name, dataset.created, dataset.meta, dataset.session,
             example.id, example.input_hash, example.task_hash, example.content
           from dataset, example, link
           where dataset.id = link.dataset_id
             and link.example_id = example.id
             and dataset.name = 'teaching_reviews_sentences-abe';""").fetchall()
    mengyuan_annotations = conn.execute(
        """select dataset.id, dataset.name, dataset.created, dataset.meta, dataset.session,
             example.id, example.input_hash, example.task_hash, example.content
           from dataset, example, link
           where dataset.id = link.dataset_id
             and link.example_id = example.id
             and dataset.name = 'teaching_reviews_sentences-mengyuan';""").fetchall()
    sakthi_annotations = conn.execute(
        """select dataset.id, dataset.name, dataset.created, dataset.meta, dataset.session,
             example.id, example.input_hash, example.task_hash, example.content
           from dataset, example, link
           where dataset.id = link.dataset_id
             and link.example_id = example.id
             and dataset.name = 'teaching_reviews_sentences-sakthi';""").fetchall()
    jenny_annotations = conn.execute(
        """select dataset.id, dataset.name, dataset.created, dataset.meta, dataset.session,
             example.id, example.input_hash, example.task_hash, example.content
           from dataset, example, link
           where dataset.id = link.dataset_id
             and link.example_id = example.id
             and dataset.name = 'teaching_reviews_sentences-jenny';""").fetchall()

    abe_anno_df = pd.DataFrame({"annotator": ["abe" for row in abe_annotations],
                           "input_hash": [row[example_input_hash_idx]
                                          for row in abe_annotations],
                           "task_hash": [row[example_task_hash_idx]
                                         for row in abe_annotations],
                           "text": [json.loads(row[example_content_idx])['text']
                                    for row in abe_annotations],
                           "annotation":  [json.loads(row[example_content_idx])['accept']
                                           for row in abe_annotations]})
    
    mengyuan_anno_df = pd.DataFrame({"annotator": ["mengyuan"
                                                   for row in mengyuan_annotations],
                           "input_hash": [row[example_input_hash_idx]
                                          for row in mengyuan_annotations],
                           "task_hash": [row[example_task_hash_idx]
                                         for row in mengyuan_annotations],
                           "text": [json.loads(row[example_content_idx])['text']
                                    for row in mengyuan_annotations],
                           "annotation":  [json.loads(row[example_content_idx])['accept']
                                           for row in mengyuan_annotations]})
    
    sakthi_anno_df = pd.DataFrame({"annotator": ["sakthi" for row in sakthi_annotations],
                           "input_hash": [row[example_input_hash_idx]
                                          for row in sakthi_annotations],
                           "task_hash": [row[example_task_hash_idx]
                                         for row in sakthi_annotations],
                           "text": [json.loads(row[example_content_idx])['text']
                                    for row in sakthi_annotations],
                           "annotation":  [json.loads(row[example_content_idx])['accept']
                                           for row in sakthi_annotations]})
    
    jenny_anno_df = pd.DataFrame({"annotator": ["jenny" for row in jenny_annotations],
                           "input_hash": [row[example_input_hash_idx]
                                          for row in jenny_annotations],
                           "task_hash": [row[example_task_hash_idx]
                                         for row in jenny_annotations],
                           "text": [json.loads(row[example_content_idx])['text']
                                    for row in jenny_annotations],
                           "annotation":  [json.loads(row[example_content_idx])['accept']
                                           for row in jenny_annotations]})
    
    # I don't think this is the ideal way: the columns could be the same length but wrong order
    #     merged_df_first_label = pd.DataFrame({
    #     "sakthi_annotation": sakthi_anno_df_first_label["annotation"],
    #     "jenny_annotation": jenny_anno_df_first_label["annotation"],
    #     "abe_annotation": abe_anno_df_first_label["annotation"],
    #     "mengyuan_annotation": mengyuan_anno_df_first_label["annotation"]
    # }).dropna()  # Drop rows with None values if any
    all_anno_df = pd.concat([abe_anno_df, mengyuan_anno_df, sakthi_anno_df,
                             jenny_anno_df])

    anno_df = all_anno_df.pivot(index=['task_hash', 'text'],
                                columns=['annotator'], values=['annotation'])
    #print(anno_df.shape, anno_df.columns) # here columns are a multi-index, don't understand
    anno_df.columns = anno_df.columns.droplevel(0) # this fixes the multi index column
    #print(anno_df.shape, anno_df.columns) # but the index is still multi-level
    anno_df.reset_index(inplace=True) # this fixes the multi-level index
    #print(anno_df.shape, anno_df.columns) # how the df is 725x4
    return anno_df

    
def get_path() -> str:
    return __file__

def get_main_label(labels: list) -> str:
    """gets a "main" label,

    ie from a list of potentially several labels, this will return the
    single label if the list has only one element, it will return
    UNSURE if that is among the labels or if there is any unrecognized
    label like None, or it will return MIXED if there are several labels
    """
    
    if len(labels) == 1:
      return labels[0]
    elif "UNSURE" in labels:
      return "UNSURE"
    elif None in labels: # this may be something to double check
      return "UNSURE"
    elif labels == None:
      return "UNSURE"
    elif "MIXED" in labels:
      return "MIXED"
    elif len(labels) > 1: # this is when multiple labels are used but not the mixed label
      return "MIXED"
    raise Exception("No main label found")
