Problem 2: problem2.m AND mykde.m

mykmeans.m is the function asked for in part 1. It takes in data X and bandwidth h and returns the estimated density in p and it's domain in x. It works for both 1D and 2D.

problem2.m contains the code used used to test problem 2. It is segmented by comments starting at Part 2. Running it consists of 16 figures:

Figure 1-4 = kde of Part 1 at bandwidths h = {.1, 1, 5, 10}
Figure 5-8 = kde of Part 2 at bandwidths h = {.1, 1, 5, 10}
Figure 9-12 = kde of Part 3, N1 at bandwidths h = {.1, 1, 5, 10}
Figure 13-16 = kde of Part 3, N2 at bandwidths h = {.1, 1, 5, 10}

Part 2 creates N = 1000 Gaussian data with the given parameters and tests mykde, as well as graphing it with a matching histogram. It makes one figure for each bandwidth.

Part 3 does the same as part 2 with the new values for the random data. I was very confused on why it gave two data parmeters, one being the same as part 2. Thus, I only tested it on the mu = 0 and sigma = 0.2  data set.

Part 4 creates the random data matricies with the given parameters and tests them in kde. It then plots each of them on a surface for each bandwidth provided.