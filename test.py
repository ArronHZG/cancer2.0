import pandas as pd
import matplotlib.pyplot as plt
a = pd.read_csv('submit/resnet18_submit.csv')

image = a.hist()
# print(image)
image.show()
# plt.imshow(image)
# plt.show()
