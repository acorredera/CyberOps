import matplotlib.pyplot as plt
import csv

x = []
y_anomaly_score = []
y_label = []
y_likelihood = []

with open('anomaly_scores_temperature.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    next(plots, None)  # skip the headers
    for row in plots:
        x.append((row[0]))
        y_anomaly_score.append(float(row[2]))
        y_label.append(float(row[4])/4)
        y_likelihood.append(float(row[3]))

plt.plot(x,y_anomaly_score, label='anomaly_score')
plt.plot(x, y_label, label='label')
plt.plot(x,y_likelihood, label='anomaly_likelihood')

plt.xlabel('timestamp')
plt.ylabel('value')

plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()
