# isye6740-su2022-teamRand

### Team Member Names: Matt Rand, Changhong Guan
### Project Title: Optimizing Initial Launch of eVTOL Operations

## Problem Statement
Urban air mobility is a fast-emerging field that is poised to revolutionize the transportation industry. Multiple companies across the world contend to bring access to a network of eVTOL aircraft that will allow for intra-city travel for the price of an Uber. While the initial economics limit the market and feasibility of the product, as the industry grows, the price point will continue to drop and expand access to broader markets across the country and across the globe.
When considering the phased growth of an industry, it becomes necessary to determine the right market dynamics to target for initial operations that properly balances growth potential with a high probability of user adoption. The United States is the perfect incubator for such an industry, with the largest cities in America experiencing high-growth in geographically constrained regions. However, not all cities developed equally. Not only do population densities vary city to city, but access to public transportation, traffic congestion, and wealth disparities change market dynamics.
A nascent industry must pick and choose its launch markets very carefully. As such, this problem is well-suited for an optimization algorithm that can determine the ideal set of launch parameters and determine which locations boast the best opportunity for success.

## Dataset
For this project, we will be using a few different data sets to create our models: 
•	US Commute Flow Data: This data set is provided by the United States Census Bureau. The data is relevant as of 2015. We are currently in search of more up-to-date data, especially post-pandemic, but believe that many of the major metropolitan cities that we are analyzing would still follow the same general trends found in the census data. Locations such as Los Angeles or New York are already large and geographically constrained with dominant industries such that we could expect traffic patterns today to mimic the general flow from suburbs to urban centers, even though densities may have changed. We can account for this by factoring the flow rates by a delta in population densities from 2015 to 2020 to approximate the change.
•	Locations of currently available US heliports: This dataset is provided by the Federal Aviation Administration and includes the latitude and longitude of all airports and heliports in the United States. We are assuming that to save on capital expenditure, initial operations will focus on retrofitting current aviation locals in lieu of developing brand new ones. 
•	Geospatial data of Greater Los Angeles: Geospatial data allows us to accurately map the Greater Los Angeles region and represent the physical location of different population centers given their latitude and longitude.
•	Census Data of Greater Los Angeles: Census data from the Greater Los Angeles region that details population density and economic status on a regional basis.
We will also be assuming some basic parameters for our eVTOL aircraft. There are numerous industry players that are contending to disrupt the space, and independent consultants have provided market analysis that dictate typical factors that influence the design of all major industry players. We will be using Deloitte’s research to benchmark the capabilities of a typical eVTOL aircraft.
From Deloitte:

	eVTOL	Ride Hailing
Speed	150 mph 	25 mph (assuming traffic)
Price per vehicle mile	$3	$2
Carbon Footprint	0 kg CO2 / 25 miles	11.1 kg CO2/25 miles
Range	60 miles (NASA)
Assume 350 miles
Boarding time	7 minutes	1 minute
De-boarding time	2 minutes	
Cost per vehicle	Unknown	$20,000

This data allows us to create two separate models to compare in our network: the time and cost for a vehicle traveling through high-commuted areas, and the time and cost for an eVTOL operation, assuming travel to a vertiport, boarding, flight, and deboarding.

## Methodology
Our ultimate goal is to evaluate the economic viability of launching urban air mobility operations as an alternative to current ride-hailing solutions, or even personal vehicle usage. To do so, we must establish a market need by identifying the cost of travel for individuals in different geographic locations, create a comparative model for idealized urban air mobility operations, and determine the difference in cost that could be captured by the urban air mobility industry. Later phases, if we have the time and resources, may include incorporating key economic factors, such as wealth of a certain geographical location, various estimates of capital expenditure for the production of aircraft based on current pricing of Part 21 aircraft (i.e. Cessna), and ………
We will conduct our analysis for the Greater Los Angeles area, a region well-publicized in the industry as a target for initial operations (Miami, New York are also viable candidates). We corroborated this sentiment using US total commute data, where we sorted by the number of commuters for each county, to find greater LA area is the heaviest in commute traffic, and thus worth exploring.
We can begin by running a cluster analysis on the geospatial data to determine how many serviceable communities there are, how large those communities are, and map those regions to the FAA airport/heliport database to determine just how many existing locations may be able to service a cluster. We can initialize cluster centers on different combinations of vertiport locations to try and force the graph to center the clusters on those locations. This isn’t guaranteed in an unsupervised algorithm, but given that there are likely many permutations of potential clusters, this technique may help to ensure that the clusters can be deemed serviceable communities 
Given the clusters determined by the algorithm, we will import the commute data, match clusters with their commute data, and generate two data sets.
1.	Dij, distance between each pair of clusters
2.	Cij, commute count between each pair of cluster
This will allow us to create a weighted graph that ranks communities around the greater region by their typical flow patterns, i.e. which locations have the most commuters leaving vs which locations accept the most commuters? This will help to identify which locations are key for developing vertiport locations to service the area. We want to establish operations where we can capture the largest portion of commuters, but also have to factor in the range limitations of the aircraft to ensure that the edges developed are all below some maximum value. We will investigate the use of both traditional clustering and ISOMAP to determine node pairs.
Given time, we may also choose to weight the graph at this phase by the average wealth of the geographic regions we identify. This would allow our ranking process to favor areas of the Greater LA region with the most wealth which, according to Deloitte, are the most likely consumer base. Comparing volume vs wealth will be an interesting experiment, because while initial operations need a wealthier client base to generate income, the volume of individuals who may potentially be serviced might outweigh the greater-than-average participation from the general population.
Given this data, we can run a network optimization routine that optimizes over the graph determined by the clusters and commute data, given the economic factors determined by the Deloitte study. We will vary this optimization by multiple factors to establish the strongest business case possible for operations.

1.	Build clustering model using kmeans based on geospatial data (ca-places-boundaries)
2.	Compute, Dij, distance between each pair of clusters, i, j in {1,2,...K}
3.	Generate cluster label for existing veriport = heliport + airport, using knn
4.	Derive commute centroids from commute data set, and generate cluster labels for both commute_from and commute_to
5.	Compute, Cij, commute count between each pair of clusters
6.	(optional) Compute, Wi, wealth weight of each cluster
7.	Optimization
Maximize		eVTOL industry profitability 
(Avg revenue/aircraft/mile – avg cost/aircraft/mile ) X Sum(# of aircrafts at each vertiport)
(Optional) Maximize 	Carbon footprint reduction
Subject to		eVTOL aircraft one-way travel range < R
eVTOL aircraft operating miles < M
- Need speed, charging time, boarding/deboarding time
% eVTOL travel adoption by commute between each pair of clusters, which could be affected by Wi, wealth weight 
Cost structure, 
- The cost of an eVTOL aircraft
- Cost of building new vertiport 
- Cost of using existing heliport
- Cost of maintenance
Percent adoption by the commute (Assuming vertiport operations are ideal and have no limiting factors such as space, timing, etc.)
o	This is obviously an idealized condition, but necessary for establishing what the maximum potential of the business is to be independent of a specific company’s operations and implementation.
8.	Solve for optimal operating # of eVTOL aircrafts at each cluster node
9.	Determine # of vertiports needed in each cluster, considering existing veriports, and assuming each veriport operates N eVTOL aircrafts
All these factors can be varied to create business plans that revolve around actionable decisions by each industry player: growing user adoption, reducing the price of the aircraft, reducing the price/vehicle mile, capital constraints when choosing to utilize existing infrastructure/creating new infrastructure.

## Evaluation and Final Results
Our results will rely on multiple models coming together to accurately map the situation. We will evaluate our models at each stage, varying parameters to idealize performance. For example, we will utilize initial cost estimates of capital expenditure and vertiport construction to limit the number possible cluster locations. We can then vary that number until we end up with a certain number of clusters that reduces mean squared error between cluster locations and population centers until we identify the best clustering for a given initial investment. 
For our optimization routine, we will have to identify the most traveled routes based on the weights of the graph. Given those weights, we will factor in the economic model for both eVTOL operations and ride-hailing to determine which combination of locations for operations maximized profitability. Profitability will be the primary evaluation criteria of the model, and we will iterate within a given set of economic factors to maximize profitability for selected operations.
Ultimately, our report will deliver idealized business cases for various capital expenditures. Our goal is that we will have a model that can be used for many cities around the US, allowing for variation in economic factors to customize the outcome given a specific operations constraints.
References

## Commute data
https://www.census.gov/topics/employment/commuting/guidance/flows.html
CA County boundaries
https://data.ca.gov/dataset/ca-geographic-boundaries
Deloitte
https://www2.deloitte.com/us/en/insights/industry/aerospace-defense/advanced-air-mobility.html
https://www2.deloitte.com/content/dam/Deloitte/us/Documents/energy-resources/eri-advanced-air-mobility.pdf 
https://www2.deloitte.com/us/en/pages/energy-and-resources/articles/the-future-of-mobility-in-aerospace-and-defense.html 
NASA
https://www.nasa.gov/mediacast/nasa-x-urban-air-mobility 

