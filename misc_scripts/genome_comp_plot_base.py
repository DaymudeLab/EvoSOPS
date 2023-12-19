# import the matplotlib package
import matplotlib.pyplot as plt

# import the numpy package
import numpy as np

# create 2 - sample a 3-Dim array, that measures
# the summer and winter rain fall amount
summer_rain = [[3, 5, 7], [15, 17, 12, 12, 15],
			[26, 21, 15]]
winter_rain = [[16, 14, 12], [31, 20, 25, 23, 28], 
			[29, 31, 35, 41]]

# the list named ticks, summarizes or groups
# the summer and winter rainfall as low, mid
# and high
ticks = ['Low', 'Mid', 'High']

# create a boxplot for two arrays separately,
# the position specifies the location of the
# particular box in the graph,
# this can be changed as per your wish. Use width
# to specify the width of the plot
summer_rain_plot = plt.boxplot(summer_rain,
							positions=np.array(
	np.arange(len(summer_rain)))*2.0, 
							widths=0.6, showfliers=False)
# *2.0-0.35
winter_rain_plot = plt.boxplot(winter_rain,
							positions=np.array(
	np.arange(len(winter_rain)))*2.0,
							widths=0.6)
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
define_box_properties(summer_rain_plot, '#D7191C', 'GA')
define_box_properties(winter_rain_plot, '#2C7BB6', 'Theory')

# set the x label values
plt.xticks(np.arange(0, len(ticks) * 2, 2), ticks)

# set the limit for x axis
plt.xlim(-2, len(ticks)*2)

# set the limit for y axis
plt.ylim(0, 50)

# set the title
plt.title('Grouped boxplot using matplotlib')

plt.show()