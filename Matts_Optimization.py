import geopandas
import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# load LA neighborhood geo data
geojson_file_loc = "data/mapping-la-data-main/geojson/la-county-neighborhoods-v6.geojson"
geo_la_neighborhood = geopandas.read_file(geojson_file_loc)
# LA EPSG transformation
geo_la_neighborhood = geo_la_neighborhood.to_crs(3310)
geo_la_neighborhood["area"] = geo_la_neighborhood.area/2.59e+6
geo_la_neighborhood = geo_la_neighborhood.to_crs(4326)
geo_la_neighborhood['boundary'] = geo_la_neighborhood.boundary
geo_la_neighborhood['centroid'] = geo_la_neighborhood.centroid

# load LA neighborhood poplulation data
pop_la_neighborhood = pd.read_csv('data/la-neighborhood-population.csv')
# merge two dataframe together
df = geo_la_neighborhood.merge(pop_la_neighborhood, left_on='name', right_on='Neighborhood', how='inner')
del df['Neighborhood']
del df['kind']
del df['external_id']
del df['slug']
del df['set']
del df['metadata']
del df['resource_uri']
df = df.rename(columns={'Population per Sqmi':'pop_density'})
df['pop'] = round(df['pop_density'] * df['area'])

# load LA existing vertiport data
df_heliport = pd.read_csv('data/us-heliports-landing-facilities.csv')
del df_heliport['State_Name']
del df_heliport['Facility_Type']
del df_heliport['County_s_State_Post_Office_Code']
df_airport = pd.read_csv('data/us-general-aviation-airports.csv')
del df_airport['County_s_State_Post_Office_Code']
del df_airport['State_Name']
df_vertiport = pd.concat([df_heliport, df_airport])
mask = (df_vertiport['SHAPE_X'] <-117.6) & (df_vertiport['SHAPE_X'] > -119) & (df_vertiport['SHAPE_Y'] <34.8) & (df_vertiport['SHAPE_Y'] > 33.6)
df_vertiport_la = df_vertiport[mask]

from scipy.sparse import csr_matrix
df_centroid = pd.DataFrame(df[['name','centroid','pop']])
n = len(df_centroid)
df_centroid['idx'] = df_centroid.index
# Cartesian product to calculate distance between all centroid pairs
centroid_pair = df_centroid.merge(df_centroid, how='cross')
gs_centroid_pair = geopandas.GeoDataFrame(centroid_pair, geometry=centroid_pair['centroid_x'], crs='EPSG:3310')
gs_centroid_pair['distance']=gs_centroid_pair.apply(lambda x: x['centroid_x'].distance(x['centroid_y']), axis=1)
gs_centroid_pair['total_pop']=gs_centroid_pair.apply(lambda x: x['pop_x'] + x['pop_y'], axis=1)
# distance matrix
D = csr_matrix((gs_centroid_pair['distance'], (gs_centroid_pair['idx_x'], gs_centroid_pair['idx_y'])), shape=(n,n))
# population matrix
# P = csr_matrix((gs_centroid_pair['total_pop'], (gs_centroid_pair['idx_x'], gs_centroid_pair['idx_y'])), shape=(n,n))
# P = P / np.max(P)

from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
embedding = Isomap(n_neighbors=10,n_components=2)
D_transformed = embedding.fit_transform(D)

sum_of_squared_distiance=[]
K = range(1,20)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(D_transformed)
    sum_of_squared_distiance.append(kmeans.inertia_)

from pulp import *
import pandas as pd

# generate optimization table
new_results_df = pd.DataFrame(columns = ['Clusters','Percent Utilization','Total Profit','Number of Daily Flights','Number of eVTOLs Needed'])

for k in range(9,13):
    

        K = k
        kmeans_alg = KMeans(n_clusters=k, random_state=0).fit(D_transformed)
        df['label']=kmeans_alg.labels_

        df_pre_dissolve = geopandas.GeoDataFrame(df, geometry=df['boundary'],crs='EPSG:4326')
        df_dissolve = df[['geometry', 'pop', 'label']].dissolve(by='label', aggfunc='sum')
        df_dissolve = df_dissolve.reset_index()
        df_dissolve['centroid'] = df_dissolve.centroid
            
        # calculate population ij, and distance ij
        df_dissolve = df_dissolve.to_crs(3310)
        for i in df_dissolve.index:
            df_dissolve['pop_'+ str(i)] = df_dissolve.apply(lambda x: (x['pop'] + df_dissolve['pop'][i])/K, axis=1)
            df_dissolve.loc[i, 'pop_'+ str(i)] = 0
            df_dissolve['dist_'+ str(i)] = df_dissolve.apply(lambda x: x['geometry'].distance(df_dissolve.loc[i, 'geometry'])/1609.34, axis=1)
            # set percentage of population will use eVTOL as commute 
            # total commute is 5% of population, eVTOL commute is 2.5% of total commute
            #df_dissolve['eVTOL%_'+ str(i)] = j/100

        
        # assign existing vertiport label
        # find which cluster center a vertiport is close to
        def min_dist(x, k=K):
            dist = []
            for i in range(k):
                dist.append(x['to_center_'+str(i)])
            if min(dist) < 0.2: # threshold can be changed
                return dist.index(min(dist))
            else:
                # ignore vertiport that is far away to any cluster center
                return 10

        ## This is what we have to come back and change... we need to be optimizing for percent of population
        def total_commute(x, k=K):
            # P=0.5 net profit per passenger per mile
            commute = 0
            for i in range(k):
                commute += x['pop_'+ str(i)] #* x['eVTOL%_'+str(i)]
            return round(commute)


        def total_prof(x, k=K, P=0.5):
            # P=0.5 net profit per passenger per mile
            prof = 0
            for i in range(k):
                prof += x['pop_'+ str(i)] * x['dist_'+ str(i)] #* x['eVTOL%_'+str(i)]
            return round(prof * P)

        gs_vertiport = geopandas.GeoSeries.from_wkt(df_vertiport_la['WKT'])
        gdf_vertiport = geopandas.GeoDataFrame(df_vertiport_la[['WKT', 'SHAPE_X', 'SHAPE_Y']], geometry=gs_vertiport, crs="EPSG:4326")
        for i in df_dissolve.index:
            gdf_vertiport['to_center_'+str(i)] = gdf_vertiport.apply(lambda x: x['geometry'].distance(df_dissolve.loc[i, 'centroid']), axis=1)
        gdf_vertiport['cluster'] = gdf_vertiport.apply(min_dist, axis=1)

        # df_dissolve['eVTOL%'] = 0.25
        df_dissolve['existing_vertiport'] = gdf_vertiport.groupby(['cluster']).size()[:-1]
        df_dissolve['total_commute'] = df_dissolve.apply(total_commute, axis=1)
        df_dissolve['total_prof'] = df_dissolve.apply(total_prof, axis=1)
        #df_dissolve.to_csv('data/cluster.csv')
        # run optimization.py in pycharm




        #result = pd.read_csv("data/optimization_result.csv", header=0)
        #final_result_df = pd.concat([df_dissolve.iloc[:, -3:], result], axis=1)
        #final_result_df.to_csv('data/result.csv')

        #result = pd.read_csv("data/optimization_result.csv", header=0)
        #final_result_df = pd.concat([df_dissolve.iloc[:, -3:], result], axis=1)
        #final_result_df.to_csv('data/result.csv')


        #data = pd.read_csv("data/cluster.csv", header=0)
        data = df_dissolve

        # only existing vertiport and total profit columns
        clusterName = data['label']
        dataTable = data.iloc[:, -3:].values.tolist()
        print(dataTable)

        n_Ve = dict([(i, n[0]) for i, n in enumerate(dataTable)])
        total_commute = dict([(i, n[1]) for i, n in enumerate(dataTable)])
        total_prof = dict([(i, n[2]) for i, n in enumerate(dataTable)])

        prob = LpProblem('MaxProfit', LpMinimize)
        solver = getSolver('GLPK_CMD')

        # number of existing vertiport
        n_eVTOL_Vars = LpVariable.dicts('n_eVTOL', clusterName, 0, cat='Integer')
        # number of new vertiport
        n_Vn_Vars = LpVariable.dicts('n_Vn', clusterName, 0, cat='Integer')
        #Percent_utilization
        util_percent_Vars = LpVariable.dicts('util_percent', clusterName, lowBound=0, upBound=1, cat='Continuous')

        n_round = 15
        c_eVTOL = 1000000.0 ## updated to 1,000,000
        c_Ve = 200000.0
        c_Vn = 500000.0
        n_Vt = 50
        cap_eVTOL = 4.0
        cap_port = 10.0
        depreciation_1 = 1/365
        depreciation_10 = 1/365/10
        depreciation_20 = 1/365/20
        total_capital = 1*10**9

        # Objection function
        prob += lpSum([- total_prof[i]*util_percent_Vars[i] + c_eVTOL * n_eVTOL_Vars[i]*depreciation_10 + c_Ve * n_Ve[i]* util_percent_Vars[i]*depreciation_1 + c_Vn *
               n_Vn_Vars[i] * depreciation_20 for i in clusterName]), 'Total Profit'

        # Constraint
        for i in clusterName:
            prob += cap_eVTOL * n_eVTOL_Vars[i] * n_round >= total_commute[i]*util_percent_Vars[i]
            prob += cap_port* (n_Ve[i] + n_Vn_Vars[i]) >= n_eVTOL_Vars[i]
            #prob += n_Vn_Vars[i] +  n_Ve[i] <= n_Vt
            prob += n_Vn_Vars[i]*c_Vn + n_Ve[i]*c_Ve + n_eVTOL_Vars[i]*c_eVTOL <= total_capital
        
        prob.solve(PULP_CBC_CMD(msg=1))

        print()

        for var in prob.variables():
            print(str(var) + ' : ' + str(var.varValue))
        print()

        print("Total profit = $%.2f" % value(-prob.objective))
        # res_dict = {'new_vertiport': [var.varValue[:10] for var in prob.variables()],
        #             'eVTOL#': [var.varValue[10:] for var in prob.variables()]}
        # res = pd.DataFrame(res_dict)
        res = [var.varValue for var in prob.variables()]
        res_df = pd.DataFrame(columns=['new_vertiport', 'eVTOL#'])
        res_df['new_vertiport'] = res[:K]
        res_df['eVTOL#'] = res[K:2*K]
        res_df['util_precent'] = res[2*K:]
        #res_df.to_csv('data/optimization_result.csv', index=False)`
        #['Clusters','Percent Utilization','Total Profit','Number of Daily Flights','Number of eVTOLs Needed']
        results_row = pd.DataFrame([k,sum(res[2*K:]),value(-prob.objective),n_round*sum(res[K:]),sum(res[K:]) ]).T
        results_row.columns =  ['Clusters','Percent Utilization','Total Profit','Number of Daily Flights','Number of eVTOLs Needed']
    
        new_results_df = pd.concat([new_results_df,results_row])
new_results_df