# Project:  EvolvingSSOPS
# Filename: plotter.py
# Authors:  Devendra Parkar (dparkar1@asu.edu) and Joshua J. Daymude
#           (jdaymude@asu.edu).

"""
plotter: Plot figures using the results of the GA experiments.
"""

from cmcrameri.cm import batlow
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp


def plot_confband(ax, x, ymean, yerr, color=None, linestyle='-', label=''):
    """
    Plots a solid line (mean) and a shaded confidence band on the given axis.

    :param ax: a matplotlib.axes.Axes object
    :param x: a numeric array of length N representing the x-coordinates
    :param ymean: a numeric array of length N representing the means
    :param yerr: a numeric array of length N representing the stddevs
    :param color: a matplotlib.colors color
    :param linestyle: a string line style for the means line
    :param label: a string legend label for this error tube
    """
    yabove = [ymean[i] + yerr[i] for i in range(len(ymean))]
    ybelow = [ymean[i] - yerr[i] for i in range(len(ymean))]
    ax.fill_between(x, yabove, ybelow, color=color, alpha=0.2)
    ax.plot(x, ymean, linestyle, color=color, label=label)


def plot_fitness_diversity(behavior, theory_fitness=None):
    """
    Plot the population's best fitness, average fitness, and diversity over the
    evolved generations. Also show the fitness value of the stochastic approach algorithm for comparison.

    :param behavior: a string collective behavior in {'agg', 'sep'}
    :param theory_fitness: a float fitness value of the theoretical algorithm,
                           or None if not available
    """
    # Load the fitness and diversity data.
    # fitnesses = np.loadtxt(osp.join('output', 'fitness_' + behavior + '.csv'),
    #                        delimiter=',')
    # diversity = np.loadtxt(osp.join('output', 'diversity_' + behavior + '.csv'),
    #                        delimiter=',')

    fitnesses = np.loadtxt('/Users/devendra/workspace/SwarmAggregationGA/output/new_t3_seg_w65_35.csv', ndmin = 2)
    diversity = np.loadtxt('/Users/devendra/workspace/SwarmAggregationGA/output/new_t5_seg_w65_35.csv', ndmin = 2)
    balance = np.loadtxt('/Users/devendra/workspace/SwarmAggregationGA/output/new_t5_big_w65_35.csv', ndmin = 2)
    balance2 = np.loadtxt('/Users/devendra/workspace/SwarmAggregationGA/output/latest_fitness_t6_multi.csv', ndmin = 2)

    # print(fitnesses)
    # print(diversity)
    # Set up the figure.
    fig, fitness_ax = plt.subplots(figsize=(8, 6), dpi=300, tight_layout=True)
    # diverse_ax = fitness_ax.twinx()
    # balance_ax = fitness_ax.twinx()
    x = np.arange(fitnesses.shape[0]) + 1
    x_div = np.arange(diversity.shape[0]) + 1
    x_bal = np.arange(balance.shape[0]) + 1
    x_bal2 = np.arange(balance2.shape[0]) + 1
    print(x.shape)
    # Set up colors.
    fitness_color, diverse_color, balance_color, balance2_color = batlow(0.2), batlow(0.4), batlow(0.6), batlow(0.8)

    # Plot the population's best and mean/stddev fitness values per generation.
    
    fitness_ax.plot(x, fitnesses, color=fitness_color,
                    label='Rank-300')
    
    fitness_ax.plot(x_div, diversity, color=diverse_color,
                    label='Tournament-300')
    
    fitness_ax.plot(x_bal, balance, color=balance_color,
                    label='Tournament-600')
    fitness_ax.plot(x_bal2, balance2, color=balance2_color,
                    label='Tournament-Expo-400')
    # plot_confband(ax=fitness_ax, x=x, ymean=np.mean(fitnesses, axis=1),
    #               yerr=np.std(fitnesses, axis=1), color=fitness_color,
    #               linestyle=':', label='Pop. Mean Fitness')

    # Plot the population's diversity per generation.
    # diverse_ax.plot(x_div, diversity, color=diverse_color)

    # balance_ax.plot(x_bal, balance, color=balance_color)

    # Set figure and axes information and coloring.
    fitness_ax.set(xlabel='Generations', xlim=[x[0], x[-1]])
    fitness_ax.set_ylabel('Fitness', color=fitness_color)
    fitness_ax.tick_params(axis='y', colors=fitness_color)
    # diverse_ax.set_ylabel('Diversity', color=diverse_color)
    # diverse_ax.tick_params(axis='y', colors=diverse_color)


    # Create a legend, which with the twin axes requires spoofing diversity.
    # fitness_ax.plot([], [], color=diverse_color, label='Tournament-Sel')
    fitness_ax.legend(loc='right')

    # Save the figure.
    fig.savefig(osp.join('output', behavior + '_fitness_best.png'))

plot_fitness_diversity('Sep_65-35-select-comp')