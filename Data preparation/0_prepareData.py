import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

path = r'input\spb.combined.daily.txt'

df = pd.read_csv(path, sep='\t')

matplotlib.rcParams["figure.figsize"] = (18, 7)
matplotlib.rcParams.update({'font.size': 22})

df.TIME = pd.to_datetime(df['TIME'], format='%Y-%m-%d')
df.set_index(['TIME'],inplace=True)
df["CONFIRMED.spb_MAM"] = df["CONFIRMED.spb"].rolling(20, min_periods=1).mean()

df["CONFIRMED_RES"] = pd.concat([df["CONFIRMED"], df["CONFIRMED.spb_MAM"]], axis=1).max(axis=1)


#Plot 1 results #Daily confirmed
#df[["CONFIRMED", "CONFIRMED.spb", "CONFIRMED.spb_MAM"]].plot()
#######

fig, ax = plt.subplots()

df[["CONFIRMED", "CONFIRMED.spb"]].plot(style='o', ax=ax, legend="")
df[["CONFIRMED_RES"]].plot(style='--', ax=ax, linewidth=3,legend="")

plt.show()

df.to_csv(r'input\spb.combined.daily_add.txt')

### FEDOT ####
df[["CONFIRMED_RES"]].to_csv(r'input\spb_confirmed_covid.csv')

#Plot 2 results #Active cases
# fig, ax = plt.subplots()
# df[["ACTIVE"]].plot(style='o', ax=ax, legend="")
# plt.ylabel("Active cases")
# plt.show()

#Plot 3 results #Activity
# fig, ax = plt.subplots()
# df[["Yandex.ACTIVITY.points"]].plot(style='-', ax=ax, legend="")
# plt.ylabel("Yandex activity points")
# plt.show()

#Plot 4 results #Deaths
# fig, ax = plt.subplots()
# df[["DEATHS"]].plot(style='r-', ax=ax, legend="")
# plt.ylabel("COVID-related deaths")
# plt.show()

#Plot 5 results #Vac 1 dose
# fig, ax = plt.subplots()
# df[["v1.CS"]].plot(style='g-', ax=ax, legend="")
# plt.ylabel("People vaccinated")
# plt.show()


