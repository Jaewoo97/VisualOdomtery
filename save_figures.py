#!/usr/bin/env python
# coding: utf-8

# Imports
from matplotlib import pyplot as plt

# Call function saveMatcher
def saveMatcher(output,
			    matcher,
                descriptor,
				ransac


				):
	# Plot BFMatcher

	# Turn interactive plotting off
	plt.ioff()

	# Create a new figure
	plt.figure()
	plt.axis('off')
	plt.imshow(output)

	plt.imsave(fname = 'Figures/ransac_%s_%s-with-%s.png' % (ransac, matcher, descriptor),
			arr = output,
			dpi = 300)

	# Close it
	plt.close()