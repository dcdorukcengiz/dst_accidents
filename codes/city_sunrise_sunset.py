%reset -f


import numpy as np
import pandas as pd
from datetime import date, datetime, time
from suntimes import SunTimes
import pytz
import json

home_directory = "/Users/EzgilovesDoruk/Desktop/education_health/"

tz_istanbul = pytz.timezone('Europe/Istanbul')
tz_name = 'Europe/Istanbul'


with open(f'{home_directory}cities_lat_long.json', 'r') as f:
    city_long_lat = json.load(f)

city_long_lat = pd.DataFrame(city_long_lat).drop(columns = ["counties"])

city_long_lat.shape[0]
all_city_data = {}

for zz in range(city_long_lat.shape[0]): 
    long_of_interest = float(city_long_lat["longitude"].iloc[zz])
    lat_of_interest = float(city_long_lat["latitude"].iloc[zz])
    altitude = 0
    kaza_ili = int(city_long_lat["plate"].iloc[zz])
    print(city_long_lat["name"].iloc[zz])

    sun2 = SunTimes(longitude=long_of_interest, latitude=lat_of_interest, altitude=altitude)


    all_days = list(pd.date_range(start="2013-01-01",end="2020-01-01"))
    all_data_dict = {}

    for zzz in range(len(all_days)):
        day = date(all_days[zzz].year, all_days[zzz].month, all_days[zzz].day)
        temp = pd.DataFrame({
            "sunrise":     [f"{sun2.risewhere(day, tz_name).hour}:{sun2.risewhere(day, tz_name).minute}"], 
            "sunset":     [f"{sun2.setwhere(day, tz_name).hour}:{sun2.setwhere(day, tz_name).minute}"], 
            "date": day, 
            "kaza_ili": [kaza_ili]})
        all_data_dict[day] = temp


#        city_data = pd.concat([city_data, temp], ignore_index = True)
    all_city_data[kaza_ili] =  pd.concat(all_data_dict, ignore_index = True).reset_index(drop = True)

final_data = pd.concat(all_city_data, ignore_index = True).reset_index(drop = True)
final_data.to_parquet(f"{home_directory}city_sunrise_sunset_table.parquet")

