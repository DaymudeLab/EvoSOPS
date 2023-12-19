import matplotlib.pyplot as plt
import numpy as np
from cmcrameri.cm import batlow
import matplotlib.ticker as tkr     # has classes for tick-locating and -formatting

def numfmt(x, pos): # your custom formatter function: divide by 100.0
    s = '{0:1.3f}'.format(x / 1000.0)
    return s

yfmt = tkr.FuncFormatter(numfmt)    # create your custom formatter function


# create 2 - sample a 3-Dim array, that measures
# the summer and winter rain fall amount
ga_probs = {'0': [63, 1000, 1000, 500], '1': [16, 500, 1000, 1000, 8, 500, 250, 250], '2': [500, 16, 1, 1000, 125, 500, 1000, 500, 16, 500, 250, 1000], '3': [63, 2, 31, 1000, 1, 1, 1000, 500, 4, 8, 125, 1000], '4': [1, 250, 2, 4, 2, 2, 8, 4], '5': [8, 63, 8, 250]}

th_probs = {'0': [1000, 1000, 1000, 1000], '1': [167, 167, 167, 167, 167, 167, 167, 167], '2': [28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28], '3': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], '4': [1, 1, 1, 1, 1, 1, 1, 1], '5': [1, 1, 1, 1]}

ga_prob = ga_probs.values()

th_prob = th_probs.values()
# the list named ticks, summarizes or groups
# the summer and winter rainfall as low, mid
# and high
ticks = ga_probs.keys()

# create a boxplot for two arrays separately,
# the position specifies the location of the
# particular box in the graph,
# this can be changed as per your wish. Use width
# to specify the width of the plot
ga_prob_plot = plt.boxplot(ga_prob,positions=np.array(np.arange(len(ga_prob)))*2.0,widths=0.4, showfliers=False, medianprops={"linewidth": 2.5})
# *2.0-0.35
th_prob_plot = plt.boxplot(th_prob,positions=np.array(np.arange(len(th_prob)))*2.0, widths=0.6,  showfliers=False, medianprops={"linewidth": 2.5})
# *2.0+0.35
# each plot returns a dictionary, use plt.setp()
# function to assign the color code
# for all properties of the box plot of particular group
# use the below function to set color for particular group,
# by iterating over all properties of the box plot
def define_box_properties(plot_name, color_code, label):
	for k, v in plot_name.items():
		plt.setp(plot_name.get(k), color=color_code)
		
	# use plot function to draw a small line to name the legend.
	plt.plot([], c=color_code, label=label)
	plt.legend()


# setting colors for each groups
define_box_properties(ga_prob_plot, batlow(0.4), 'GA') #'#D7191C'
define_box_properties(th_prob_plot, batlow(0.6), 'Theory') #'#2C7BB6'

# set the x label values
plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)

# set the limit for x axis
plt.xlim(-2, len(ticks)*2)

# set the limit for y axis
# plt.ylim(-100, 1100)
plt.semilogy(base=2)
plt.gca().yaxis.set_major_formatter(yfmt)

plt.xlabel(r"Current # of Neighbors $e$") # $e(\tau) - e(\sigma)$ , color=batlow(0.1))
plt.ylabel('Move Probability') #, color=batlow(0.1))

# set the title
plt.title('Aggregation Genomic Composition')

plt.savefig('output/Agg_gene_comp.png', dpi=300, bbox_inches='tight')
# plt.show()