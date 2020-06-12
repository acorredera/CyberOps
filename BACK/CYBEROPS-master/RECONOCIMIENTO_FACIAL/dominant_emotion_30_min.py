import csv
import pandas as pd
import datetime
import os

PATH = os.getcwd() #/home/marta/PycharmProjects/RECONOCIMIENTO_FACIAL

data = []

with open(PATH+'/Webcam_photos/emotions.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', lineterminator='\n')
    for row in reader:
        data.append(row)

header=data[0]
data.pop(0)

df = pd.DataFrame(data, columns=header)
# print(df)

df['timestamp'] = [datetime.datetime.strptime(d, "%Y-%m-%d_%H_%M_%S") for d in df["timestamp"]] # this line converts the string object in Timestamp object
df['date'] = [datetime.datetime.date(d) for d in df['timestamp']]# extracting date from timestamp
df['time'] = [datetime.datetime.time(d) for d in df['timestamp']]# extracting time from timestamp

# unique_date = pd.unique(df["date"])
# print(unique_date)

# period = datetime(0,30)

initial_time = []
final_time = []
minimum_time = min(df["timestamp"])
date0 = minimum_time

for date in df["timestamp"]:
    max_time =  date0 + datetime.timedelta(minutes = 30)
    if date <= max_time:
        final_time.append(max_time)
        initial_time.append(date0)
    else:
       date0 = date
       max_time = date0 + datetime.timedelta(minutes = 30)
       final_time.append(max_time)
       initial_time.append(date0)

df['time_start_period'] = initial_time
df['time_final_period'] = final_time

# print(df)

df_dominant_emotion = df.groupby(["time_start_period", "time_final_period", "emotion"]).count()
df_dominant_emotion = df_dominant_emotion.reset_index()
# df_dominant_emotion.sort()
# df_dominant_emotion = df_dominant_emotion.sort_values('time', ascending=False)

df_count = pd.DataFrame(df_dominant_emotion)
df_count["count_n_emotion"] = df_count["time"]
df_count = df_count[["time_start_period", "time_final_period", "emotion", "count_n_emotion"]]
# print(df_count)

# row_0 = 0
# emotion_0 = ""
# dominant_emotion = []
#
# for row in df_count["time_start_period"].index():
#     if df_count["time_start_period"][row] == df_count["time_start_period"][row_0]:
#         if df_count["time"][row] > df_count[row_0]
# df.groupby('a').count()

df_count_total_date = df.groupby("time_start_period").count()
df_count_total_date = df_count_total_date.reset_index()
df_count_total_date["total_count"] = df_count_total_date["time"]
df_count_total_date = df_count_total_date[["time_start_period", "total_count"]]
# print(df_count_total_date)

# merge df_count and df_count_total_date

df_emotion = df_count.merge(df_count_total_date, on="time_start_period")
# df_emotion = df.merge(df_count_total_date, on="time_start_period", how='inner')
# print(df_emotion)

percentage_list = []
#
for row in df_emotion["count_n_emotion"].index:
    percentage = round((df_emotion["count_n_emotion"][row]/float(df_emotion["total_count"][row]))*100, 2)
    percentage_list.append(percentage)

df_emotion["percentage"] = percentage_list
df_emotion = df_emotion[["time_start_period", "time_final_period", "emotion", "percentage"]]

print(df_emotion)
# print(type(df_emotion))

df_emotion.to_csv((PATH + '/Webcam_photos/dominant_emotion_30.csv'), index=False)
