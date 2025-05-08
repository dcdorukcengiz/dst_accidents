%reset -f

# Import Meteostat library and dependencies
from datetime import datetime
import matplotlib.pyplot as plt
from meteostat import Point, Daily
import json 
import pandas as pd
import requests
import time

home_directory = "/Users/EzgilovesDoruk/Desktop/education_health/"

with open(f'{home_directory}cities_lat_long.json', 'r') as f:
    city_long_lat = json.load(f)

city_long_lat = pd.DataFrame(city_long_lat).drop(columns = ["counties"])

all_data = {}
for zz in range(city_long_lat.shape[0]):
    print(zz)
    time.sleep(3)
    print(city_long_lat["name"].iloc[zz])
    lat = float(city_long_lat["latitude"].iloc[zz])
    lon = float(city_long_lat["longitude"].iloc[zz])
    plaka = city_long_lat["plate"].iloc[zz]
    # Set time period
    start = datetime(2013, 1, 1)
    end = datetime(2019, 12, 31)


    url = f"https://api.opentopodata.org/v1/aster30m?locations={lat},{lon}"
    r = requests.get(url)

    json_data = r.json()
    alt = json_data["results"][0]["elevation"]

    location = Point(lat, lon, alt)

    # Get daily data for 2018
    data = Daily(location, start, end)
    data = data.fetch()
    data2 = data.reset_index().assign(**{"kaza_ili": plaka})
    all_data[plaka] = data2

final_data = pd.concat(all_data, ignore_index=True)
final_data.to_parquet(f"{home_directory}weather_data.parquet")
