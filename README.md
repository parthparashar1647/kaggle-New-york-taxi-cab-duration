# kaggle-New-york-taxi-cab-duration

THE STEPS ARE AS FOLLOWS:->

The Problem


New York is riddled with one-ways, small side streets, and an almost incalculable amount of pedestrians at any given point in time. Not to mention the amount of cars/motorcycles/bicycles clogging up the roads. Combine this with a mad rush to get from point A to point B, and you'll find yourself late for whatever you need to be on time for.

The solution to getting from A to B when living in a city like New York (without losing your mind) is easy: take a taxi/Uber/Lyft/etc. You don't need to stress about the traffic or pedestrians and you have a moment to do something else, like catch up on emails. Although this sounds simple enough, it doesn't mean you'll get to your destination in time. So you need to have your driver take the shortest trip possible. By shortest, we're talking time. If a route A is X kilometers *longer*, but gets you there Y minutes *faster* than route B would, rather take that one.

To know which route is the best one to take, we need to be able to predict how long the trip will last when taking a specific route. Therefore, *the goal of this playground competition is to predict the the duration of each trip in the test data set, given start and end coordinates.*


The Libraries & Functions


Using Python 3.6.1, import the following libraries. Note the use of `%matplotlib inline`, allowing the display of graphs inline in iPython Notebook.

Documentation

Scikit-Learn
Pandas
Numpy
XGBoost
Seaborn
I used Scikit-Learn (or sklearn) for a few of the machine learning operations that was carried out. Pandas is used for data manipulation. Numpy is the fundamental package for scientific computation in Python. XGBoost is the classification algorithm used to make the final predictions. Seaborn is a nice tool for data visualisation built on top of matplotlib.

2. Data Summary
Getting a statistical summary of the data is also quite easy. This is where the `describe` function is used.


3 Data Preparation
My initial thoughts on this is that from the pickup_datetime field we should be extracting hour of day and day of week and day of month. This seems logical since peak hour traffic and weekend vs non-weekend days could have a mayor effect on travel times. Similar to this there could be seasonality in the travel times to be observed between the different months of the year. Think, for example, of the effect New York's winter months might have on travel time. If the roads are wet/icy you're less likely to drive very fast, so regardless of which route you're taking, you'll take a bit longer to reach your destination.

An interesting variable and its effects to potentially explore is the `passenger_count`. One might argue that an increased number of passengers could result in a few scenarios. It could, for example, result in increased stops along the way, which ultimately extends the time from A to B (note how we're not given the number of passengers at the start and end of the trip - my thinking is that if we were to know the start and end number of passengers in a trip, the data between drops would be split into separate entries, with the last entry in a collection of rows ending with 0 passengers). Also, from a purely physical point of view, the higher the number of passengers, the heavier the vehicle, the slower the vehicle might move. Although I'm fairly certain the effect of this is neglible.

Same with `vendor` and `store_and_fwd_flag`. It is possible that looking into it, we find that there's a difference between vendor 1 and vendor 2, and that either one of the two might be corrupting the "shortest route information" because the vendor's employees are less efficient in finding the best routes through New York. This, however, I also find highly unlikely, but it is an alternative to explore (at least look into it and rule it out difinitively). As for `store_and_fwd_flag` - not having a connection to the server for a particular route to be indicative of a few things. For example, If upon inspection it is discovered that there is a strong correlation between slow trip times and server disconnects, it could be used as another feature in the training model to predict what time a particular route could take.

Getting down to it however, the `_lattitude` and `_longitute` variables is where I think the most value lies. The options here are to cluster them into "neighborhoods" as well as find the distance and directions between coordinates.

4.Trip Duration Clean-up


As we noted earlier there are some outliers associated with the `trip_duration` variable, specifically a 980 hour maximum trip duration and a minimum of 1 second trip duration. I've decided to exclude data that lies outside 2 standard deviations from the mean. It might be worthwhile looking into what effect excluding up to 4 standard deviations would have on the end-results.

5. Latitude and Longitude Clean-up


Looking into it, the borders of NY City, in coordinates comes out to be:

city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85) 

Comparing this to our `train.describe()` output we see that there are some coordinate points (pick ups/drop offs) that fall outside these borders. So let's limit our area of investigation to within the NY City borders.

6. Data Visualisation and Analysis
These next steps involve looking at the data visually. Often you'll discover looking at something significant as a graph rather than a table (for example) will give you far greater insight into its nature and what you might need to do to work with it. Of course, the opposite could also be considered true, so don't neglect the first section we went through.

7. Applying these functions to both the test and train data, we can calculate the haversine distance which is the great-circle distance between two points on a sphere given their longitudes and latitudes. We can then calculate the summed distance traveled in Manhattan. And finally we calculate (through some handy trigonometry) the direction (or bearing) of the distance traveled. These calculations are stored as variables in the separate data sets. The next step I decided to take is to create neighourhods, like Soho, or the Upper East Side, from the data and display this.

8. XGBoost - Training the Model and Testing the Accuracy
As mentioned you can play with the different parameters of the XGBoost algorithm to tweak the model's outcome. So below is a short, but very nice, way of itterating through model parameters to tweak the model. So it's implementation is simple: just uncomment the code and run the kernel. Again, refer to the [documentation for XGBoost](http://xgboost.readthedocs.io/en/latest/python/python_intro.html "XGBoost Documentation") to understand what each parameter does and how it alters the behaviour of the training process.

9.So from top to bottom we see which features have the greatest affect on trip duration. It would make logical sense that distance has the greatest affect. The further you travel, the longer it'll take. The rest of the features follow a similar logic in why it's ranked the way it is.


The final step before submission is to make our predictions using the trained model.
