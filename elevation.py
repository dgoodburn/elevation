__author__ = 'dangoodburn'

from numpy import cos, sin, sqrt, pi
import numpy as np
import pandas as pd
from pandas.io import sql
import requests
import instance
from sqlalchemy import create_engine

database = instance.getDatabase()
engine = create_engine(database)

Earth = {
    'radius' : 6371000
}


##################################### GET LOCATIONS ######################################

def send_request(locations):
# returns locations requested

    address = ""

    VarKey = instance.getAPIkey()
    link0 = "https://maps.googleapis.com/maps/api/elevation/json?locations="
    link1 = "&key="

    address += link0

    for i in locations:
        address += str(i[0])
        address += ","
        address += str(i[1])
        address += "|"

    address = address[:-1]
    address += link1
    address += VarKey

    print address

    return requests.get(address).json()


def compile_locations(position):
    # splits locations into groups to send to api

    locations = []

    for i in range(position[0][1] - position[0][0] + 1):
        for j in range(position[1][1] - position[1][0] + 1):
            locations.append([i + position[0][0], j + position[1][0]])

    return locations


def get_locations(locations):

    string_locations = []

    locations_json = send_request(locations)
    print locations_json

    for i in range(len(locations_json['results'])):
        string_locations.append([locations_json['results'][i]['location']['lat'],locations_json['results'][i]['location']['lng'],locations_json['results'][i]['elevation']])

    insert_into_table(string_locations)


def query_elevations():

    locations = []
    position = [[0.1, 89.9], [-179.9, 179.9]]  # [-180, 180]
    limit = 125
    limit_count = 0

    for i in range(int(round(position[0][1]*10 - position[0][0]*10 + 1, 0))):
        for j in range(int(round(position[1][1]*10 - position[1][0]*10 + 1, 0))):
            locations.append([1.0 * i/10 + position[0][0], 1.0 * j/10 + position[1][0]])
            if limit_count < limit:
                limit_count += 1
            elif limit_count >= limit:
                try_get_locations(locations)
                limit_count = 0
                locations = []
            else:
                print "error getting locations"
                break
    if limit_count != 0:
        try_get_locations(locations)


def query_elevations2(positions):

    locations = []

    limit = 100
    limit_count = 0

    for i in positions:

        locations.append([i[0], i[1]])
        if limit_count < limit:
            limit_count += 1
        elif limit_count >= limit:
            try_get_locations(locations)
            limit_count = 0
            locations = []
        else:
            print "error getting locations"
            break
    if limit_count != 0:
        try_get_locations(locations)


exceptions = []
def try_get_locations(locations):

    global exceptions
    try:
        get_locations(locations)
    except:
        exceptions += locations


def insert_into_table(locations):

    query = """ INSERT INTO TBLelevation (latitude, longitude, elevation) VALUES """

    for i in locations:
        query += "(" + str(i[0]) + ", " + str(i[1]) + ", " + str(i[2]) + "),"

    query = query[:-1]
    print query
    sql.execute(query, engine)





import time

timeiteration = 0


def timerfunc(func, *args):

    global timeiteration

    start = time.time()
    timeiteration += 1

    func(*args)

    print timeiteration, func.__name__, "%.5f" % (time.time() - start) + " sec"

    #return returnvalue


def find_missing_locations():

    query = """

    SELECT
        longitude,
        latitude
      FROM TBLelevation

    """

    df_locations = pd.read_sql(query, engine)
    df_locations['missing'] = 1

    factor = 10

    df_all = pd.DataFrame({ 'latitude' : range(int(-89.9*factor), int(89.9*factor), 1), 'join' : 1})
    df_all['latitude'] /= factor

    df_join = pd.DataFrame({ 'longitude' : range(int(-179.9*factor), int(179.9*factor), 1), 'join' : 1})
    df_join['longitude'] /= factor

    df_all = df_all.merge(df_join, on="join")
    del df_all['join']

    df = df_all.merge(df_locations, on=['latitude', 'longitude'], how='left')
    df = df[df['missing'] != 1]
    del df['missing']

    list = df.values.tolist()

    #print list

    query_elevations2(list)
    print exceptions


def find_visibility():

    DistBtwnLat = 112000
    Earth = 6371000
    PIvar = pi * 2.0 / 360.0
    MaxDistance = 224000
    maxDegrees = MaxDistance / DistBtwnLat

    #df_lat = pd.DataFrame({'latitude':arange(-180,180), 'merge':1})
    #df_long = pd.DataFrame({'longitude':arange(-360,360), 'merge':1})

    query = """

    SELECT
        longitude,
        latitude,
        elevation,
        1 as merge
      FROM TBLelevationV1
      WHERE elevation > 0
      ORDER BY latitude, longitude

    """

    df_original = pd.read_sql(query, engine)

    df_original.latitude = df_original.latitude.astype(np.float16)
    df_original.longitude = df_original.longitude.astype(np.float16)
    df_original.elevation = df_original.elevation.astype(np.int16)

    df_valley = df_original[['latitude', 'longitude', 'elevation', 'merge']]
    df_mountain = df_original[df_original.elevation > 4000][['latitude', 'longitude', 'elevation', 'merge']]
    del df_original['merge']

    split = 10
    splitlong = 10
    counter = 0

    for k in range(split):
        for j in range(splitlong):
            i = k
            df = df_valley[(df_valley.latitude >= (-90 + i * 180/split)) &
                           (df_valley.latitude < (-90 + (i+1) * 180/split)) &
                           (df_valley.longitude >= (-180 + j * 360/splitlong)) &
                           (df_valley.longitude < (-180 + (j+1) * 360/splitlong))]
            print (1.0 * i*split + j)/(split*splitlong)
            df = df.merge(df_mountain[(df_mountain.latitude >= (-90 + i * 180/split - maxDegrees)) &
                                      (df_mountain.latitude < (-90 + (i+1) * 180/split + maxDegrees)) &
                                      (df_mountain.longitude >= (-180 + j * 360/splitlong - maxDegrees)) &
                                      (df_mountain.longitude < (-180 + (j+1) * 360/splitlong + maxDegrees))],
                          on="merge", copy=False)

            if df.empty==False:
                del df['merge']

                df = df.ix[df['elevation_y'] - df['elevation_x'] > 4000]

                counter += 1
                if counter == 10:
                    counter = 0

                    df['visible'] = ((1.0 / (cos(df['latitude_x'].astype(np.float64) * PIvar) *
                                             cos(df['latitude_y'].astype(np.float64) * PIvar) *
                                             cos(df['longitude_x'].astype(np.float64) * PIvar -
                                                 df['longitude_y'].astype(np.float64) * PIvar)  - 1)) * \
                        Earth) + df['elevation_x']

                    df = df.sort(['visible'], ascending=False)
                    df = df.iloc[0:1000]

                try:
                    final_df = final_df.append(df)
                except:
                    final_df = df

        dist = (1 - ((MaxDistance*1.0 / Earth)**2) / 2)

        try:
            final_df = final_df[dist < \
                cos(df['latitude_x'].astype(np.float64) * PIvar) *
                                             cos(df['latitude_y'].astype(np.float64) * PIvar) *
                                             cos(df['longitude_x'].astype(np.float64) * PIvar -
                                                 df['longitude_y'].astype(np.float64) * PIvar)]
        except:
            pass

    try:
        df = final_df
    except:
        pass

    df['visible'] = ((1.0 / (cos(df['latitude_x'].astype(np.float64) * PIvar) *
                                             cos(df['latitude_y'].astype(np.float64) * PIvar) *
                                             cos(df['longitude_x'].astype(np.float64) * PIvar -
                                                 df['longitude_y'].astype(np.float64) * PIvar)  - 1)) * \
                        Earth) + df['elevation_x']

    df = df[df.visible > 0]

    df = df.merge(df_original, left_on=['latitude_y','longitude_y'], right_on=['latitude','longitude'],
                  copy=False)[['latitude_x','longitude_x','latitude_y','longitude_y',
                               'elevation_x','elevation_y','visible']]
    df['visible'] = df.elevation_y - df.visible
    df = df[df.visible > 0]
    df = df.sort(['visible'], ascending=False)
    return df


def plot_locations():

    import numpy as np
    import matplotlib.pyplot as plt

    query = """

    SELECT T1.longitude, T1.latitude, T1.elevation
    FROM TBLelevationV1 as T1;

    """
    df = pd.read_sql(query, engine)

    fig, ax = plt.subplots()

    min_latitude = min(df['latitude'])
    min_longitude = min(df['longitude'])
    width = int(max(df['latitude']) - min_latitude)
    height = int(max(df['longitude']) - min_longitude)

    image = np.zeros((width, height))

    def seas(value):  # sets water squares to -1000

        if value < 0:
            return -1000
        else:
            return value

    for i in df.iterrows():

        image[int(i[1][1]-min_latitude-1)][int(i[1][0]-min_longitude-1)] = seas(i[1][2])

    ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax.set_title('dropped spines')
    ax.invert_yaxis()

    plt.show()

#plot_locations()


def plot_3d():

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    query = """

    SELECT
        round(cos(latitude * 2.0 / 360 * PI()) * cos(longitude * 2.0 / 360 * PI()),2) As X,
        round(sin(latitude * 2.0 / 360 * PI()),2) As Y,
        round(cos(latitude * 2.0 / 360 * PI()) * sin(longitude * 2.0 / 360 * PI()),2) as Z,
        elevation
    FROM TBLelevationV1

    """
    df = pd.read_sql(query, engine)
    df.ix[df.elevation<0, 'elevation'] = -1000
    df.ix[df.elevation>=0, 'elevation'] += 1000000

    threedee = plt.figure().gca(projection='3d')
    threedee.scatter(df.X, df.Y, df.Z, c=df.elevation*1000)
    plt.show()

#plot_3d()

def sql_litedb():
    # load in memory sqlite database from mysql

    import sqlite3 as db
    engine2=create_engine('sqlite:///temp.db', echo=True)

    query = "SELECT latitude, longitude, elevation FROM TBLelevationV1"

    df = pd.read_sql(query, engine)

    df.to_sql('TBLelevation', engine2, if_exists='replace', index=False)

    ind = "CREATE UNIQUE INDEX lat_long_el_index ON TBLelevation(latitude, longitude, elevation);"

    conn = db.connect('temp.db')
    c = conn.cursor()
    c.execute(ind)

    query = "SELECT * FROM TBLelevationV1 LIMIT 10"

    results = c.execute(query).fetchall()
    print results

    #setup_procedure()
    #df = pd.read_sql(query, engine2)
    #print df.head()


def get_files():

    #engine2=create_engine('sqlite://', echo=True)
    query = "SELECT * FROM TBLelevationV1"

    df = pd.read_sql(query, engine)
    return df

#get_files().to_csv('elevationV1.csv')

def setup_procedure():

    import sqlite3 as db
    #engine2=create_engine('sqlite://', echo=True)

    conn = db.connect('temp.db')
    c = conn.cursor()

    DistBtwnLat = 112000
    Earth = 6371000
    Radius = 2.0
    PIvar = (2.0 / 360 * 3.1415)
    NumNeeded = 1000
    #MaxHeight = ( "SELECT MAX(elevation) FROM TBLelevationV1");
    MaxHeight = 6000
    #MaxDistance = Earth*SQRT(2-2/(MaxHeight/Earth+1))
    MaxDistance = (Earth*sqrt(2-2/((MaxHeight-3000)/Earth+1)))
    MaxDistance = (6371000*sqrt(2-2/((6000-3000)/6371000+1)))
    ElevationDiff = 4000

    query = """

        SELECT
          S1.LO1,
          S1.LA1,
          S1.EL1,
          S2.LO1,
          S2.LA1,
          S2.EL1,
          ((SIN(S1.LA1 * PIvar) * SIN(S2.LA1 * PIvar) + COS(S1.LA1 * PIvar) *
                COS(S2.LA1 * PIvar) * COS(S2.LO1 * PIvar - S1.LO1 * PIvar))*
           (S2.EL1/Earth+1)-1)*Earth-S1.EL1 AS VISIBLE
        FROM
          (SELECT
             T1.longitude AS LO1,
             T1.latitude  AS LA1,
             T1.elevation AS EL1
           FROM TBLelevation AS T1
           WHERE T1.elevation > 0 AND T1.elevation < (MaxHeight-ElevationDiff)) AS S1
          CROSS JOIN
          (SELECT
             T1.longitude AS LO1,
             T1.latitude  AS LA1,
             T1.elevation AS EL1
           FROM TBLelevation AS T1
           WHERE T1.elevation > ElevationDiff) AS S2
        WHERE ABS(S1.LA1 - S2.LA1) < (MaxDistance / DistBtwnLat)
          AND S2.EL1 - S1.EL1 > ElevationDiff
          AND (1 - (pow((MaxDistance / Earth), 2) / 2)) <
              (SIN(S1.LA1 * PIvar) * SIN(S2.LA1 * PIvar) + COS(S1.LA1 * PIvar) *
              COS(S2.LA1 * PIvar) * COS(S2.LO1 * PIvar - S1.LO1 * PIvar))
          AND ((SIN(S1.LA1 * PIvar) * SIN(S2.LA1 * PIvar) + COS(S1.LA1 * PIvar) *
                COS(S2.LA1 * PIvar) * COS(S2.LO1 * PIvar - S1.LO1 * PIvar))*
               (S2.EL1/Earth+1)-1)*Earth-S1.EL1 > ElevationDiff

        ORDER BY VISIBLE DESC
      ;


    """ #% (table, i, i)

    query = """

        SELECT
          S1.LO1,
          S1.LA1,
          S1.EL1,
          S2.LO1,
          S2.LA1,
          S2.EL1,
          ((SIN(S1.LA1 * (2.0 / 360 * 3.1415)) * SIN(S2.LA1 * (2.0 / 360 * 3.1415)) + COS(S1.LA1 * (2.0 / 360 * 3.1415)) *
                COS(S2.LA1 * (2.0 / 360 * 3.1415)) * COS(S2.LO1 * (2.0 / 360 * 3.1415) - S1.LO1 * (2.0 / 360 * 3.1415)))*
           (S2.EL1/6371000+1)-1)*6371000-S1.EL1 AS VISIBLE
        FROM
          (SELECT
             T1.longitude AS LO1,
             T1.latitude  AS LA1,
             T1.elevation AS EL1
           FROM TBLelevation AS T1
           WHERE T1.elevation > 0 AND T1.elevation < (6000-4000)) AS S1
          CROSS JOIN
          (SELECT
             T1.longitude AS LO1,
             T1.latitude  AS LA1,
             T1.elevation AS EL1
           FROM TBLelevation AS T1
           WHERE T1.elevation > 4000) AS S2
        WHERE ABS(S1.LA1 - S2.LA1) < ((6371000*sqrt(2-2/((6000-3000)/6371000+1))) / 112000)
          AND S2.EL1 - S1.EL1 > 4000

        ORDER BY VISIBLE DESC
      ;


    """

    #sql.execute(query, engine2)
    results = c.execute(query).fetchall()
    return results


#sql_litedb()
#setup_procedure()
#get_files()

