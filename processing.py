import sqlite3

con = sqlite3.connect("prodigy.sentence-level.20240605.db")
cur = con.cursor()


query_1 = """
select * from dataset, example, link 
where dataset.id = link.dataset_id 
and link.example_id = example.id 
and dataset.name = 'teaching_reviews_sentences-abe';
"""

query_2 = """
select * from dataset, example, link 
where dataset.id = link.dataset_id 
and link.example_id = example.id 
and dataset.name = 'teaching_reviews_sentences-mengyuan';
"""

query_3 = """
select count(*) from dataset, example, link 
where dataset.id = link.dataset_id 
and link.example_id = example.id 
and dataset.name like 'teaching_reviews_sentences-%';
"""

for row in cur.execute(query_2):
	print(row)


# cur.execute(query_2)
# rows = cur.fetchall()
# for row in rows:
#     print(row)



# Closing the Connection
con.close()
