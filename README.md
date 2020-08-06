# CurvatureRadii
 Find the curvature radii of the largest oil blob in a single channel along the oil surface in 2D images.

 The oil blob in the pore has different curvature radii at different places. This program computes the curvature radii at different places in the oil blob.

 1. find the oil-water interface
 2. use adjacent points to fit circles
 3. visualize the radii of curvature

Examples can be found in the findCurvatureRadii.py file, and also in runexp1.py and run20191107ppt.py, though more than needed functions are provided.

This program provides a radius of curvature computing function, but the one in findCurvatureRadii2.py is recommended, which is from the scipy cookbook website.
