from matplotlib import pyplot as plt

returns =  [143.1, 140.6, 112.5, 63.9, 35.8]
true_returns = [57.6, 58.0, 56.9, 56.9, 34.5]

plt.scatter(returns, true_returns, color='red')
plt.xlabel("Returns")
plt.ylabel("True Returns")
# both range from 0 to 200
plt.xlim(0, 150)
plt.ylim(0, 150)

# add a diagonal line
plt.plot([0, 200], [0, 200], color='black', linestyle='--')

plt.show()