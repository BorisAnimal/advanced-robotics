from MSA import *

x1, y1, z1, d1 = xScatter, yScatter, zScatter, dScatter

from VJM import *

x2, y2, z2, d2 = xScatter, yScatter, zScatter, dScatter

plot_deflection(x1, y1, z1, d1 - d2)

