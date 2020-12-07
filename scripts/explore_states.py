import sys
from learning.state_exploration.counter import Counter

from matplotlib import pyplot as plt

from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

if __name__ == "__main__": # for multiprocessing
    s = './results/state_counts/test.json'

    # c = Counter(s)

    # anim = FuncAnimation(c.fig, c.start, init_func=c._init_plots, frames=10000)

    # # anim.save('assets/state_intersection.gif', writer='imagemagick')
    # plt.show()

    c = Counter(s)

    anim = FuncAnimation(c.fig, c.start, init_func=c._init_plots, frames=1000, interval=50)

    anim.save('assets/state_intersection.gif', writer='imagemagick')
    c.fig.savefig('assets/state_intersection_final2.png')
    # plt.show()
    # c.start(1000)