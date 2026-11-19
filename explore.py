import matplotlib.pyplot as plt

from bikes_data import make_rentals

df = make_rentals()

# Average rentals for each hour
hourly = df.groupby("hour")["rentals"].mean()

# Individual observations and the hourly mean
plt.scatter(df["hour"], df["rentals"], alpha=0.06, s=8)
plt.plot(hourly.index, hourly.values, linewidth=2)

plt.title("Bike Rentals by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Bikes Rented")
plt.xticks(range(0, 24, 2))
plt.tight_layout()
plt.show()
