import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import requests
import pandas as pd
import csv

# Abrir gephi:  ./bin/gephi en su directorio

filenames = ["variables_iniciales.csv"]

filenames_with_path = [os.path.join("/home/marta/Descargas/Documentos_texto/", fn) for fn in filenames]

# these texts have underscores ('_') that indicate italics; remove them.
raw_texts = []

for fn in filenames_with_path:
    with open(fn) as f:
        text = f.read()
        text = text.replace(' ', '_')  # remove underscores (italics)
        raw_texts.append(text)

vectorizer = CountVectorizer(input='content', lowercase=True, stop_words="english")

dtm = vectorizer.fit_transform(raw_texts)

vocab = np.array(vectorizer.get_feature_names())
# print(len(vocab))
subset_title=["id"]
variables = set(vocab)- set(subset_title)
# print(len(variables))

source = []  #starting_node
weight = []
target = [] #ending_node
relation = []

for word in variables:

    obj = requests.get("http://api.conceptnet.io/c/en/%s" % (word)).json() #to search more than 1 word: write with "_" (hello_world)

    for i in obj["edges"]:
        source.append(i["start"]["label"])
        weight.append(i["weight"])
        target.append(i["end"]["label"])
        relation.append(i["rel"]["@id"][3:])

table = {"source": source, "weight": weight, "target": target, "relation": relation}
# print (table)

df = pd.DataFrame(table)
print (df)

df.to_csv("/home/marta/Descargas/Documentos_texto/variables_edges.csv", sep=';', index=False)

starting_nodes = set(source)
unique_starting_nodes = list(starting_nodes)
# print(unique_starting_nodes)
#
ending_nodes = set(target)
unique_ending_nodes = list(ending_nodes)
# print("List of unique numbers : ", unique_ending_nodes)
#
unique_nodes = np.unique(df[['source', 'target']].values)
# print(unique_nodes)

# csv.writer(unique_nodes, delimiter=";")
#
# # wtr = csv.writer(open ("/home/marta/Descargas/Documentos_texto/nodes.csv", 'w'), delimiter=';', lineterminator='\n')
# # for x in unique_nodes : wtr.writerow ([x])
#
f = open('/home/marta/Descargas/Documentos_texto/variables_nodes.csv', 'w')

with f:
    fnames = ['Id', 'Label']
    writer = csv.DictWriter(f, fieldnames=fnames, delimiter=";", lineterminator='\n')
    writer.writeheader()
    for x in unique_nodes: writer.writerow({'Id' : x, "Label":x})