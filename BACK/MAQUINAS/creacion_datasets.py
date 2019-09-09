import glob
import os

PATH = os.getcwd()

# # folders = glob.glob(PATH+"/NAB/data/*")
# # print(folders)
# # for i in folders:
# #     files = glob.glob(i+"/*.csv")
#
# folder_list=[]
# dict={}
#
# for root, dirs, files in os.walk(PATH+"/NAB/data/"): #carpeta data
#     for folder in dirs:
#         folder_list.append(folder)
#
# for name in folder_list:
#     for root, dirs, files in os.walk(PATH+"/NAB/data/%s" %name):
#         dict[name]=[]
#         name_files=[]
#         for file in files:
#             if file.endswith('.csv'):
#                 name_files.append(file)
#             dict[name] = name_files
# #     #     print(folder)
# #     # for file in files:
# #     #     if file.endswith('.csv'):
# #     #         print(file)
#
# print(folder_list)
# print(dict)

import csv
import json

PATH = os.getcwd()

with open(PATH+"/NAB/labels/raw/AL_labels_v0.8.json") as f1, open(PATH+"/NAB/labels/raw/AL_labels_v1.0.json") as f2, open(PATH + "/NAB/labels/raw/CB_labels_v0.8.json") as f3, open(PATH+"/NAB/labels/raw/CB_labels_v1.0.json") as f4, open(PATH+"/NAB/labels/raw/known_labels_v1.0.json") as f5, open(PATH+"/NAB/labels/raw/SA_labels_v0.8.json") as f6, open(PATH+"/NAB/labels/raw/SA_labels_v1.0.json") as f7, open(PATH+"/NAB/labels/combined_labels.json") as f8:
    AL_labels_8 = json.load(f1)
    AL_labels_1 = json.load(f2)
    CB_labels_8 = json.load(f3)
    CB_labels_1 = json.load(f4)
    known_labels_1 = json.load(f5)
    SA_labels_8 = json.load(f6)
    SA_labels_1 = json.load(f7)
    combined_labels = json.load(f8)

list_label_files = [AL_labels_8, AL_labels_1, CB_labels_8, CB_labels_1, known_labels_1, SA_labels_8, SA_labels_1, combined_labels]


def labeled_files():
    for root, dirs, files in os.walk(PATH+"/NAB/data/"): #carpeta data

        for file in files:

            if file.endswith('.csv'):

                folder= root.split("/")[-1]
                if not os.path.exists("Labeled_data/"):
                    os.makedirs("Labeled_data/")
                with open(root+"/%s" %file, 'r') as f, open('Labeled_data/%s_labeled.csv' %file[:-4], 'w') as f_out:
                    reader = csv.reader(f)
                    writer = csv.writer(f_out)
                    count = 0
                    for row in reader:

                        if count==0:
                            writer.writerow((row[0], row[1], 'label'))
                        else:
                            row[0] = row[0] + ".000000"
                            label_value = 0
                            key = "".join([folder, "/", file])

                            for label_file in list_label_files: #Compare with all label files
                                if key in label_file:
                                    if row[0] in label_file[key]:
                                        label_value=1
                                        continue

                            writer.writerow((row[0], row[1], label_value))
                        count += 1

labeled_files()