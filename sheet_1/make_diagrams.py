import matplotlib.pyplot as plt
import pandas as pd

adam_4 = pd.read_excel("./4_adam.xlsx", index_col=0)
adam_4 = adam_4.T
print(adam_4.head())
adam_4.plot(subplots=True)
plt.title("Adam with batch size 4")
plt.tight_layout()
plt.savefig("adam_4.png")

adam_16 = pd.read_excel("./16_adam.xlsx", index_col=0)
adam_16 = adam_16.T
print(adam_16.head())
adam_16.plot(subplots=True)
plt.title("Adam with batch size 16")
plt.tight_layout()
plt.savefig("adam_16.png")

adam_64 = pd.read_excel("./64_adam.xlsx", index_col=0)
adam_64 = adam_64.T
print(adam_64.head())
adam_64.plot(subplots=True)
plt.title("Adam with batch size 64")
plt.tight_layout()
plt.savefig("adam_64.png")

sgd_64 = pd.read_excel("./64_sgd.xlsx", index_col=0)
sgd_64 = sgd_64.T
print(sgd_64.head())
sgd_64.plot(subplots=True)
plt.title("SGD with batch size 64")
plt.tight_layout()
plt.savefig("sgd_64.png")
