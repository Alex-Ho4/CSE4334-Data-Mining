Problem 1: problem1.m AND mykmeans.m

mykmeans.m is the function asked for in part 1. It takes data X, number of clusters k, and the intial cluster centers c. It returns the vector of clusters who's row corrisponds to a data point's row. It also prints out the number of iterations and the cluster centers.

problem1.m contains the code I used to test the code, segmented by comments. Part 1 sets up the data that will be used, using mvnrnd and the given mus and sigmas. These are put into R1 to be used in part 1 and R2 to be used in part 2. It also creates a color cell array to simply be used when plotting.

Part 2 calls mykmeans for the given data set R1, the number of clusters 2, and the given starting centers. mykmeans.m will return a vector of clusters, which corrispond to the row of the given data. It is then plotted with a different color for each cluster.

Part 3 does what part 2 does, just with the the data set R2, 4 clusters, and the four given initial centers.