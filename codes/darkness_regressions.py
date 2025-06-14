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



combined_accidents_cleaner = pd.read_parquet("data/combined_accidents_cleaner_table.parquet")
weather_data = (pd.read_parquet("data/weather_data.parquet")[["time", "tavg", "kaza_ili"]].drop_duplicates().
            assign(**{"date": lambda x: x["time"].dt.date}).
            assign(**{"kaza_ili": lambda x: x["kaza_ili"].astype(int)})
)

federal_holidays = pd.read_parquet(f"data/federal_holidays_turkey_table.parquet").rename(columns = {"date": "kazatarihi_date"})


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
    assign(**{"full_light_share": lambda x: x["hourly_date"].case_when(
        [
            ( (x["hour"] >  x["sunrise_hour"] + 1) & (x["hour"] < x["sunset_hour"] - 1)        , 60 ), 
            ( (x["hour"] <  x["sunrise_hour"] + 1)                                         , 0 ), 
            ( (x["hour"] >  x["sunset_hour"] - 1)                                          , 0 ), 
            ( (x["hour"] ==  x["sunrise_hour"] + 1)                                        , 60 -  x["sunrise_minute"] ), 
            ( (x["hour"] ==  x["sunset_hour"] - 1)                                         , x["sunset_minute"] ), 
            (pd.Series(True) , 99999)
            
        ]
    )}).
    assign(**{"dim_light_share": lambda x: x["light_share"] - x["full_light_share"]}).
    assign(**{"dim_light_share": lambda x: (x["dim_light_share"]/60.0).astype(float)}).    
    assign(**{"full_light_share": lambda x: (x["full_light_share"]/60.0).astype(float)}).    
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
    assign(**{"year": lambda x: x["hourly_date"].dt.isocalendar().year}).
    assign(**{"day_of_the_week": lambda x: x["hourly_date"].dt.dayofweek}).
    assign(**{"monthly_year": lambda x: x["hourly_date"].dt.strftime("%Y-%m")}).
    assign(**{"month": lambda x: x["hourly_date"].dt.month}).
    assign(**{"week": lambda x: x["hourly_date"].dt.isocalendar().week}).
    assign(**{"day": lambda x: x["hourly_date"].dt.day}).
    assign(**{"year_half": lambda x: x["month"] <= 6})
)

balanced_data_w_accidents = (balanced_data_w_accidents.
    merge(weather_data[["date", "kaza_ili", "tavg"]].drop_duplicates(), on= ["date", "kaza_ili"], how = "left")
)



balanced_data_w_accidents['panel_id'] = balanced_data_w_accidents.groupby(["hourly_date_sans_year", "kaza_ili"]).ngroup()
balanced_data_w_accidents['day_of_the_week_hour_city'] = balanced_data_w_accidents.groupby(["day_of_the_week", "kaza_ili", "hour", "month"]).ngroup()
balanced_data_w_accidents['day_of_the_week_hour_week_city'] = balanced_data_w_accidents.groupby(["day_of_the_week", "kaza_ili", "week", "hour"]).ngroup()

balanced_data_w_accidents['panel_id2'] = balanced_data_w_accidents.groupby(["week", "day_of_the_week",  "hour", "kaza_ili"]).ngroup()



balanced_data_w_accidents["panel_id"].nunique() * balanced_data_w_accidents["kaza_ili"].nunique()

balanced_data_w_accidents_w_holidays = (balanced_data_w_accidents.
    merge(federal_holidays.assign(**{"date": lambda x: pd.to_datetime(x["kazatarihi_date"]).dt.date})[["date", "holiday"]].drop_duplicates(), on = ["date"], 
    how = "left").
    assign(**{"holiday": lambda x: x["holiday"].fillna('no_holiday')}).
    assign(**{"day_squared": lambda x: x["day"]**2}).
    assign(**{"tavg_squared": lambda x: x["tavg"] ** 2})

)

balanced_data_w_accidents_w_holidays_2017_onwards = balanced_data_w_accidents_w_holidays.query("year > 2016").copy()
balanced_data_w_accidents_w_holidays["year_cat"] = balanced_data_w_accidents_w_holidays["year"].astype('category')
balanced_data_w_accidents_w_holidays_2017_onwards["year_cat"] = balanced_data_w_accidents_w_holidays_2017_onwards["year"].astype('category')
fitted_model = pf.fepois("total_accidents ~ light_share  + tavg | panel_id + year + day_of_the_week" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday'"))
fitted_model.vcov("HC1").summary()
fitted_model.summary()

fitted_model = pf.fepois("total_accidents ~ light_share  + tavg | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday'"))
fitted_model.vcov("HC1").summary()

fitted_model = pf.fepois("total_accidents ~ light_share  + tavg + tavg_squared | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday'"))
fitted_model.vcov("HC1").summary()


#Estimation is possible for two reasons: 1- minute differences in sunrise and sunset; 2-Same ISO weeks and day of the week having different sunrise and sundown hours; 3- earlier daylight saving.
fitted_model = pf.fepois("total_accidents ~ light_share | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and month != 3 and month != 4 and month != 10 and month != 11"))
fitted_model.vcov("HC1").summary()

fitted_model = pf.fepois("total_accidents ~ light_share | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and (month < 3 or month > 11)"))
fitted_model.vcov("HC1").summary()

#Wrong mindset: Where does the estimation come from? Not from a ton of places here. It does not come from same hour, same place, different level of light exposure.
fitted_model = pf.fepois("total_accidents ~ light_share | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and (month > 4 and month < 10) and year > 2016"))
fitted_model.vcov("HC1").summary()
fitted_model = pf.fepois("total_accidents ~ light_share | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and (month > 4 and month < 10) and year < 2016"))
fitted_model.vcov("HC1").summary()

#Correct mindset: Same hour, same place. Majority of estimation come from daylight saving stuff
fitted_model = pf.fepois("total_accidents ~ light_share | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and (month < 3 or month > 11)"))
fitted_model.vcov("HC1").summary()
fitted_model.vcov({"CRV1": "year_half + kaza_ili"}).summary()

fitted_model = pf.fepois("total_accidents ~ full_light_share + dim_light_share | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and (month < 3 or month > 11)"))
fitted_model.vcov({"CRV1": "year_half + kaza_ili"}).summary()

fitted_model = pf.fepois("total_accidents ~ full_light_share + dim_light_share  + year:C(kaza_ili) | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and (month < 3 or month > 11) and hour > 12"))
fitted_model.vcov({"CRV1": "year_half + kaza_ili"}).summary()


fitted_model = pf.fepois("total_accidents ~ light_share  + year:C(kaza_ili) + year:C(hour) + tavg + tavg_squared | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and (month < 3 or month > 11) and hour > 12"))
fitted_model.vcov({"CRV1": "year_half + kaza_ili"}).summary()

fitted_model = pf.fepois("total_accidents ~ light_share  + year:C(kaza_ili) + year:C(hour) + tavg + tavg_squared | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and (month < 3 or month > 11) and hour < 12"))
fitted_model.vcov({"CRV1": "year_half + kaza_ili"}).summary()
#This is placebo-like (is it though?! feels like it only shows mornings are not that important when it comes to light share)
fitted_model = pf.fepois("total_accidents ~ light_share  + year:C(kaza_ili) + year:C(hour) + tavg + tavg_squared | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and (month > 4 or month < 10) and hour < 12"))
fitted_model.vcov({"CRV1": "year_half + kaza_ili"}).summary()

fitted_model = pf.fepois("total_accidents ~ light_share  + year:C(kaza_ili) + year:C(hour) + tavg + tavg_squared | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and (month > 4 and month < 10) and hour > 12"))
fitted_model.vcov({"CRV1": "year_half + kaza_ili"}).summary()

fitted_model = pf.fepois("total_accidents ~ light_share  + year:C(kaza_ili) + year:C(hour) + tavg + tavg_squared | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and (month > 4 and month < 10) and hour < 12"))
fitted_model.vcov({"CRV1": "year_half + kaza_ili"}).summary()


#The following takes ages. Not sure why
fitted_model = pf.fepois("total_accidents ~ light_share  + year:C(kaza_ili) + year:C(hour) | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and (month > 4 or month < 10)"))
fitted_model.vcov({"CRV1": "year_half + kaza_ili"}).summary()

#Seems like before noon light impact is limited (and slightly positive even)
fitted_model2 = pf.fepois("total_accidents ~ light_share  + year:C(kaza_ili) + year:C(hour) + tavg + tavg_squared | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and hour < 12"))
fitted_model2.vcov({"CRV1": "year_half + kaza_ili"}).summary()

fitted_model2 = pf.fepois("total_accidents ~ light_share  + year:C(kaza_ili) + year:C(hour) + tavg + tavg_squared | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and hour > 12"))
fitted_model2.vcov({"CRV1": "year_half + kaza_ili"}).summary()

fitted_model2 = pf.fepois("total_accidents ~ light_share  + year:C(kaza_ili) + year:C(hour) + tavg + tavg_squared | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and hour < 12"))
fitted_model2.vcov({"CRV1": "year_half + kaza_ili"}).summary()


#This is good. 
fitted_model = pf.fepois("total_accidents ~ full_light_share + dim_light_share  + year:C(kaza_ili) + year:C(hour)  | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday'"))
fitted_model.vcov({"CRV1": "year_half + kaza_ili"}).summary()


fitted_model = pf.fepois("total_accidents ~ light_share  + year:C(kaza_ili) + year:C(hour) | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and (month > 4 or month < 10) and hour < 12"))
fitted_model.vcov({"CRV1": "year_half + kaza_ili"}).summary()


fitted_model = pf.fepois("total_accidents ~ light_share | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday'"))
fitted_model.vcov({"CRV1": "year_half + kaza_ili"}).summary()







#The morning effect is strange. It is positive: more light more accidents
fitted_model = pf.fepois("total_accidents ~ light_share | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and  hour < 12 and (month < 3 or month > 11)"))
fitted_model.vcov("HC1").summary()
fitted_model.vcov({"CRV1": "year_half + kaza_ili"}).summary()

fitted_model = pf.fepois("total_accidents ~ light_share + year:C(kaza_ili) | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.assign(**{"year2": lambda x: x["year"]**2}).query("holiday == 'no_holiday' and  hour < 12 and (month > 4 or month < 10)"))
fitted_model.vcov({"CRV1": "year + kaza_ili"}).summary()



#The afternoon effect is good. It is negative: more light less accidents
fitted_model = pf.fepois("total_accidents ~ light_share | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and hour > 12  and (month < 3 or month > 11) "))
fitted_model.vcov("HC1").summary()
fitted_model.vcov({"CRV1": "hour + kaza_ili"}).summary()

#If we focus on summer months, as expected, the estimates are very imprecise
fitted_model = pf.fepois("total_accidents ~ light_share | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and hour < 12  and (month > 4 and month < 10) "))
fitted_model.vcov({"CRV1": "year_half + kaza_ili"}).summary()

# If we look at the entire year, since the majority of variation comes from daylight saving, the estimates follow earlier findings. 
fitted_model = pf.fepois("total_accidents ~ light_share | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and hour < 12  "))
fitted_model.vcov({"CRV1": "year_half + kaza_ili"}).summary()

fitted_model = pf.fepois("total_accidents ~ light_share | panel_id2 + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and hour > 12  "))
fitted_model.vcov({"CRV1": "year_half + kaza_ili"}).summary()

fitted_model = pf.fepois("total_accidents ~ light_share  | panel_id2 + year_cat" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and month != 3 and month != 4 and month != 10 and month != 11"))
fitted_model.vcov("HC1").summary()
fitted_model.summary()

fitted_model = pf.fepois("total_accidents ~ light_share + day + day_squared | day_of_the_week_hour_city + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday'"))
fitted_model.vcov("HC1").summary()
fitted_model.vcov({"CRV1": "kaza_ili"}).summary()

fitted_model = pf.feols("total_accidents ~ light_share | panel_id + year + day_of_the_week_hour_city" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday'"))
fitted_model.summary()


zz = 7

fitted_model = pf.fepois("total_accidents ~ light_share | day_of_the_week_hour_week_city + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and hour == @zz and month != 3 and month !=4 and month != 10 and month != 11"))
fitted_model.vcov({"CRV1": "kaza_ili"}).summary()

fitted_model = pf.feols("total_accidents ~ light_share | day_of_the_week_hour_city + year" , data = balanced_data_w_accidents_w_holidays.query("holiday == 'no_holiday' and year < 2016 and hour == @zz and month != 3 and month !=4 and month != 10 and month != 11"))
fitted_model.vcov("HC1").summary()


