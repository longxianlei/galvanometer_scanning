# Galvanometer Scanning
All plot function to plot the scan route of samples.
All code about a galvanomirror scanning, route planning, sampling, data analysis and feedback control.

Reform the code blocks. 
1. We can specify the start and end points. 
2. Using the pre-processed samples sequence to further solve the TSP problem.
3. The pre-processed method includes X, Y axis based process.

We have 3 kind of scanning routes planning algorithms,
which includes ortools based solver, 2-opt solver, basic sort algorithm.
Besides, we can specify the start and end points of the route planning algorithm.

![scan mode](scan_mode.png)
We have Snake scan, Raster scan, and other scan based algorithm.