# chroptiks

## Description
chroptiks is a Python package that offers advanced plotting utilities, making it easier to create complex and informative visualizations. It extends the functionality of libraries like matplotlib and scipy, providing a user-friendly interface for a variety of plotting needs.

## Installation

To install chroptiks, run:

```bash
git clone https://github.com/cagostino/chroptiks.git
cd chroptiks
python setup.py install
```
## Usage

Here's how you can use your_package_name in your projects:

from chroptiks.plotting_utils import hist2d, hist1d, scatter, plot3d, plotbar

# Example usage
# hist2d, hist1d, scatter, plot3d, plotbar are now ready to be used as per their defined functionalities and plots are generated through each of their plot methods. For example:

```python

import numpy as np
from chroptiks.plotting_utils import hist2d

x = np.linspace(0,100, 10000)
y = np.linspace(0,100, 10000)*(-3)+5

hist2d.plot(x,y)
```

## Features

2D Histograms (hist2d): Easily create 2D histograms for data analysis.
1D Histograms (hist1d): Simplify the process of creating and customizing 1D histograms.
Scatter Plots (scatter): Enhanced functionality for scatter plot creation.
3D Plots (plot3d): Intuitive tools for 3D data visualization.
Bar Charts (plotbar): Quick and customizable bar chart creation.

## Requirements

Python libraries: matplotlib, numpy, scipy

Other: LaTeX. If you don't want to use LaTeX in the plotting, edit the plotting_utils file and comment out the line : `plt.rc('text', usetex=True)`
