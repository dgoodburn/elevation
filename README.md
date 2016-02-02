# ELEVATION

### Background

I always wondered when you look towards the horizon and see a mountain, just how high a mountain like Everest would appear. Knowing that the land around Mt. Everest is much above sea level though, I wondered if the relative height of it was as relatively massive.
As a result, I wanted to see if I could figure out what was the highest visible mountain from anywhere on Earth.

### Data

In order to figure this out, I needed to know the altitudes at each place on Earth. To obtain this, I found the Google Maps Elevation API which allows you to return about 500,000 altitudes each day (for free). However, the surface area of the Earth is approximately 500M square kilometres, meaning that one data point from one day's worth of data would be 1,000 square kilometres which seemed much to big. I set up a script to pull the data for a couple weeks until I got about 6M data points (~80 sq km's each). This was still big but reasonable. I input this data into a MySQL database.

### Formula for visible height

In order to calculate the visible height, I needed to determine the height that is hidden behind the curvature of the Earth. Per the following diagram:

<img src="https://github.com/dgoodburn/elevation/blob/master/elevation_pic.png" width="288">

## RESULTS

Click the following link to see full notebook:
http://nbviewer.jupyter.org/github/dgoodburn/elevation/blob/master/Elevations.ipynb
