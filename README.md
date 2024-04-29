# chroptiks

## Description
chroptiks is a Python package that offers advanced plotting utilities, making it easier to create complex and informative visualizations. It extends the functionality of libraries like matplotlib and scipy, providing a user-friendly interface for a variety of plotting needs.


## Requirements

Python libraries: matplotlib, numpy, scipy

## Installation

To install chroptiks, run:

```bash
pip install chroptiks
```

or if you want to install from source:
```bash
git clone https://github.com/cagostino/chroptiks.git
cd chroptiks
python setup.py install
```


## Features

---2D Histograms (hist2d): Easily create 2D histograms for data analysis.

---1D Histograms (hist1d): Simplify the process of creating and customizing 1D histograms.

---Scatter Plots (scatter): Enhanced functionality for scatter plot creation.

---3D Plots (plot3d): Intuitive tools for 3D data visualization.

---Bar Charts (plotbar): Quick and customizable bar chart creation.


## Example usage
hist2d, hist1d, scatter, plot3d, plotbar are now ready to be used as per their defined functionalities and plots are generated through each of their plot methods. For example:

```python

import numpy as np
from chroptiks.plotting_utils import hist2d

x = np.linspace(-1,1, 100000)+np.random.normal(0,.1, size=100000)
y = x**(3)+np.random.normal(0, 0.3, size=100000)
z =  np.random.normal(size=100000)

hist2d.plot(x,y,nx=200,ny=200)

hist2d.plot(x,y,nx=40, ny=40, ccode = z, ccodename='Z', xlabel='X', ylabel='Y')

hist2d.plot(x,y,nx=200,ny=200,bin_y=True, size_y_bin=0.1, xlabel='X', ylabel='Y', percentiles=False)

from chroptiks.plotting_utils import hist1d

hist1d.plot(z, range=(-2,2), bins=100, xlabel='Z', ylabel='Counts')

hist1d.plot(z, range=(-2,2), bins=100, xlabel='Z', ylabel='Counts', normed=True)

hist1d.plot(z, range=(-4,4), bins=100, xlabel='Z', ylabel='Cumulative Count', cumulative=True)



from chroptiks.plotting_utils import scat

#scat
scat.plot(x,z)

#scat with color-code
scat.plot(x,y, ccode=z, color=None, edgecolor=None, vmin=-0.5, vmax=0.5)
#note if you use ccode to color the points, you must set color to None, and I would advise you to set edgecolor to None as well or else each will have outlines.

#scat with y-binning
scat.plot(x, y, bin_y=True, size_y_bin=0.1, percentiles=True, xlabel='X', ylabel='Z', aspect='auto')



```
