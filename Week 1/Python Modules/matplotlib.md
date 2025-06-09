### Installing MathPlotLib
```
pip install matplotlib
```
### Importing MatplotLib
```
import matplotlib.pyplot as plt
```

## Diffrent Type Of Plots in MathPlotLib

### 1. Line Plot
Used to show trends over time or continuous data.
```
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [10, 20, 15, 25]

plt.plot(x, y)
plt.title("Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

### 2. Bar Plot
Used to compare categories of data.
```
x = ['A', 'B', 'C', 'D']
y = [5, 7, 3, 8]

plt.bar(x, y)
plt.title("Bar Plot")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.show()
```
### 3. Histogram
Shows the distribution of a dataset.
```
import numpy as np

data = np.random.randn(1000)

plt.hist(data, bins=20)
plt.title("Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()
```
### 4. Scatter Plot
Shows relationships or correlation between two variables.
```
x = [1, 2, 3, 4, 5]
y = [5, 4, 2, 6, 7]

plt.scatter(x, y)
plt.title("Scatter Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
```
### 5. Pie Chart
Shows proportions of a whole.
```
labels = ['Apple', 'Banana', 'Cherry', 'Date']
sizes = [30, 25, 20, 25]

plt.pie(sizes, labels=labels, autopct='%1.1f%%')
plt.title("Pie Chart")
plt.show()
```
### 6. Box Plot
Used to display the distribution, median, and outliers of a dataset.
```
data = [7, 15, 13, 10, 22, 30, 5, 8, 15]

plt.boxplot(data)
plt.title("Box Plot")
plt.ylabel("Values")
plt.show()
```
### 7. Area Plot (Stack Plot)
Shows cumulative totals over time.
```
days = [1, 2, 3, 4]
eating = [3, 4, 2, 4]
sleeping = [8, 7, 8, 9]

plt.stackplot(days, eating, sleeping, labels=['Eating', 'Sleeping'])
plt.title("Stack Plot")
plt.xlabel("Days")
plt.ylabel("Hours")
plt.legend(loc='upper left')
plt.show()
```

## Addition Resources
[MathPlotLib Documentation](https://matplotlib.org/stable/index.html)<br>
[Video Resources](https://youtu.be/qErBw-R2Ybk?si=eZ6WUInYrTqwPy5i)
