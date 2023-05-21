import matplotlib.pyplot as plt
import numpy as np



fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
X = np.arange(4)
ax.set_xlabel('K')
ax.set_ylabel('MSE')
ax.bar(0.25, color = 'b', width = 0.25, height=0.5, label = 'validation')
# ax.bar(X + 0.25, (1,1), color = 'g', width = 0.25, label = 'training')
ax.legend()
plt.savefig('pics/barplot.png')
plt.show()