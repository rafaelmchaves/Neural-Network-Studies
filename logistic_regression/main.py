# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets


plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    a = np.array([2, 3, 4])
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
    print(a)
    print(np.zeros((3, 4)))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
