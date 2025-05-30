%reset -f

import numpy as np
import pandas as pd
import janitor
import matplotlib.pyplot as plt
import matplotlib as mp
import pyfixest as pf
from datetime import time, datetime
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.ticker as ticker 
import matplotlib.dates as mdates


home_directory = "/Users/EzgilovesDoruk/Desktop/education_health/"
combined_accidents_cleaner = pd.read_parquet(f"{home_directory}combined_accidents_cleaner_table.parquet")


federal_holidays = pd.read_parquet(f"{home_directory}federal_holidays_turkey_table.parquet").rename(columns = {"date": "kazatarihi_date"})


#Balanced data. The panel axis is city-hourly and the time is year.
start_date = '2013-01-01 00:00:00'  # Include the time (00:00:00 for the start of the day)
end_date = '2019-12-31 23:00:00' # Include the time (23:00:00 for the end of the day)

sunrise_sunset_hours = pd.read_parquet(f"{home_directory}city_sunrise_sunset_table.parquet")

all_hours = pd.DataFrame({"hourly_date": pd.date_range(start_date, end_date, freq="H")})
all_cities = combined_accidents_cleaner[["kaza_ili"]].drop_duplicates()
balanced_data = all_hours.merge(all_cities, how = 'cross')
balanced_data = (balanced_data.
    assign(**{"date": lambda x: x["hourly_date"].dt.date}).
    merge(sunrise_sunset_hours, how = "inner", on = ["date", "kaza_ili"]).
    assign(**{"sunrise_datetime": lambda x: pd.to_datetime(pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d") + "_" + x["sunrise"] + ":00", format='%Y-%m-%d_%H:%M:%S')}).
    assign(**{"sunset_datetime": lambda x: pd.to_datetime(pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d") + "_" + x["sunset"] + ":00", format='%Y-%m-%d_%H:%M:%S')}).
    assign(**{"sunrise_hour": lambda x: x["sunrise_datetime"].dt.hour}).
    assign(**{"sunset_hour": lambda x: x["sunset_datetime"].dt.hour}).
    assign(**{"sunrise_minute": lambda x: x["sunrise_datetime"].dt.minute}).
    assign(**{"sunset_minute": lambda x: x["sunset_datetime"].dt.minute}).
    assign(**{"hour": lambda x: x["hourly_date"].dt.hour}).
    assign(**{"light_share": lambda x: x["hourly_date"].case_when(
        [
            ( (x["hour"] >  x["sunrise_hour"]) & (x["hour"] < x["sunset_hour"])        , 60 ), 
            ( (x["hour"] <  x["sunrise_hour"])                                         , 0 ), 
            ( (x["hour"] >  x["sunset_hour"])                                          , 0 ), 
            ( (x["hour"] ==  x["sunrise_hour"])                                        , 60 -  x["sunrise_minute"] ), 
            ( (x["hour"] ==  x["sunset_hour"])                                         , x["sunset_minute"] ), 
            (pd.Series(True) , 99999)
            
        ]
    )}).
    assign(**{"light_share": lambda x: (x["light_share"] / 60.0).astype(float)})

)


combined_accidents_hourly = (combined_accidents_cleaner.
    assign(**{"hourly_date": lambda x: x["kazatarihi_full"].dt.floor('H')}).
    groupby(["hourly_date", "kaza_ili"]).
    agg(total_accidents = ('kazaid', 'count')).reset_index()
)



balanced_data_w_accidents = (balanced_data.
    merge(combined_accidents_hourly, on = ["hourly_date", "kaza_ili"], how = "left").
    assign(**{"total_accidents": lambda x: x["total_accidents"].fillna(0)}).
    assign(**{"hourly_date_sans_year": lambda x: x["hourly_date"].dt.strftime('%m-%d-%H')}).
    assign(**{"monthly_date_sans_year": lambda x: x["hourly_date"].dt.strftime('%m-%d')}).
    assign(**{"year": lambda x: x["hourly_date"].dt.year}).
    assign(**{"day_of_the_week": lambda x: x["hourly_date"].dt.dayofweek}).
    assign(**{"month": lambda x: x["hourly_date"].dt.strftime("%m")})
)

balanced_data_w_accidents['panel_id'] = balanced_data_w_accidents.groupby(["hourly_date_sans_year", "kaza_ili"]).ngroup()
balanced_data_w_accidents['day_of_the_week_hour_city'] = balanced_data_w_accidents.groupby(["day_of_the_week", "kaza_ili", "month"]).ngroup()


balanced_data_w_accidents["panel_id"].nunique() * balanced_data_w_accidents["kaza_ili"].nunique()

balanced_data_w_accidents_w_holidays = (balanced_data_w_accidents.
    merge(federal_holidays.assign(**{"date": lambda x: pd.to_datetime(x["kazatarihi_date"]).dt.date})[["date", "holiday"]].drop_duplicates(), on = ["date"], 
    how = "left").
    assign(**{"holiday": lambda x: x["holiday"].fillna('no_holiday')})

)

fitted_model = pf.fepois("total_accidents ~ light_share | panel_id + year + day_of_the_week" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and year > 2016"))
fitted_model.vcov("HC1").summary()


fitted_model = pf.fepois("total_accidents ~ light_share | panel_id + year + day_of_the_week" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday'"))
fitted_model.vcov("HC1").summary()
fitted_model.vcov({"CRV1": "kaza_ili"}).summary()

pf.feols("total_accidents ~ light_share | panel_id + year + day_of_the_week_hour_city" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday'"))

balanced_data_w_accidents.columns