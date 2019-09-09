import csv
import pandas as pd

_INPUT_DATA_FILE = "temperature_train.csv"

# Temperature among 13-26 G

df = pd.read_csv(_INPUT_DATA_FILE)
df=df[2:]

min=13
max=26
Nmin=30
Nmax=50

normalized_celsius=[]
# print(df)
for i in df["c1"]:
    i=float(i)
    celsius= (i-32)/1.8
    normalized=Nmin+(((celsius-min)*(Nmax-Nmin))/(max-min))
    normalized_celsius.append(normalized)

df["c1"]=normalized_celsius
# print(df)

df.to_csv("celsius_temperature_train.csv", sep=',', index=False)

# with open(_INPUT_DATA_FILE) as f:


# with open(_INPUT_DATA_FILE) as f:
#     temperature=sum(1 for line in f)-3
#     csvWriter = csv.writer(f)
#     csvWriter.writerow(new_data)
