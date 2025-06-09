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


combined_accidents_cleaner = pd.read_parquet(f"data/combined_accidents_cleaner_table.parquet")


num_weeks_diff = 2
combined_accidents_cleaner_w_dst = (combined_accidents_cleaner.
assign(**{"accident_year_month": lambda x: x["kazatarihi_full"].dt.strftime("%Y-%m")}).
assign(**{"two_weeks_before_after_change": lambda x: 
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2013-03-31') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2013-03-31'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2013-10-27') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2013-10-27'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2014-03-30') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2014-03-30'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2014-10-26') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2014-10-26'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2015-03-29') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2015-03-29'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2015-11-08') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2015-11-08'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2016-03-27') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2016-03-27'))  
    
    }).
assign(**{"two_weeks_after_change": lambda x: 
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2013-03-31') & (x["kazatarihi_full"] >= '2013-03-31'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2013-10-27') & (x["kazatarihi_full"] >= '2013-10-27'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2014-03-30') & (x["kazatarihi_full"] >= '2014-03-30'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2014-10-26') & (x["kazatarihi_full"] >= '2014-10-26'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2015-03-29') & (x["kazatarihi_full"] >= '2015-03-29'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2015-11-08') & (x["kazatarihi_full"] >= '2015-11-08'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2016-03-27') & (x["kazatarihi_full"] >= '2016-03-27'))  
    
    }).
assign(**{"placebo_two_weeks_before_after_change": lambda x: 
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2017-04-02') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2017-04-02'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2017-10-29') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2017-10-29'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2018-04-01') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2018-04-01'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2018-10-28') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2018-10-28'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2019-03-31') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2019-03-31'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2019-11-03') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2019-11-03'))      
    }).
assign(**{"placebo_two_weeks_after_change": lambda x: 
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2017-04-02') & (x["kazatarihi_full"] >= '2017-04-02'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2017-10-29') & (x["kazatarihi_full"] >= '2017-10-29'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2018-04-01') & (x["kazatarihi_full"] >= '2018-04-01'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2018-10-28') & (x["kazatarihi_full"] >= '2018-10-28'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2019-03-31') & (x["kazatarihi_full"] >= '2019-03-31'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2019-11-03') & (x["kazatarihi_full"] >= '2019-11-03'))  
    
    }).
assign(**{"accident_time": lambda x: x["kazatarihi_full"].dt.time}).
assign(**{"accident_date": lambda x: x["kazatarihi_full"].dt.date})

)


all_time_changes = pd.DataFrame({"year": list(range(2013,2016)) + list(range(2017, 2020)), 
                                "t0" : ["2013-10-27", "2014-10-26", "2015-11-08", "2017-10-29", "2018-10-28", "2019-11-03"]}
).assign(**{"t0": lambda x: pd.to_datetime(x["t0"]).dt.date})
                                

federal_holidays = pd.read_parquet("data/federal_holidays_turkey_table.parquet").rename(columns = {"date": "accident_date"}).assign(**{"accident_date": lambda x: pd.to_datetime(x["accident_date"]).dt.date   })
combined_accidents_cleaner_w_dst_w_holidays = combined_accidents_cleaner_w_dst.merge(federal_holidays, on = "accident_date", how = "left")
zoomed_in_data_agg = (combined_accidents_cleaner_w_dst_w_holidays.query("two_weeks_before_after_change == True and kazatarihi_full.dt.month > 6").
    merge(all_time_changes, left_on = ["kazayili"], right_on = ["year"] ).
    assign(**{"day_distance": lambda x: (pd.to_datetime(x["accident_date"]) - pd.to_datetime(x["t0"])).dt.days})
)


zoomed_in_data_agg.info()

all_estimates = pd.DataFrame()
for start_hour in range(0, 24, hour_range):
#for start_hour in range(16, 18, 1):

    end_hour = start_hour + hour_range

    start_time = time(start_hour,0,0)
    if end_hour == 24:
        end_time = time(23,59,59)
    else: 
        end_time = time(end_hour,0,0)


    ############         All Accidents        #######################
    zoomed_in_data_agg = (combined_accidents_cleaner_w_dst.query("two_weeks_before_after_change == True and kazatarihi_full.dt.month > 6")
    .query("accident_time <= @end_time and accident_time >= @start_time")
    .assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).groupby(["year", "kaza_ili", "two_weeks_after_change"]).
    agg(total_accidents = ('kazaid', 'count')).reset_index())

    placebo_zoomed_in_data_agg = (combined_accidents_cleaner_w_dst.query("placebo_two_weeks_before_after_change == True and kazatarihi_full.dt.month > 6").
    query("accident_time <= @end_time and accident_time >= @start_time"). 
    assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).groupby(["year", "kaza_ili", "placebo_two_weeks_after_change"]).
    agg(total_accidents = ('kazaid', 'count')).reset_index())

    to_balance_data = (combined_accidents_cleaner_w_dst[["kaza_ili"]].drop_duplicates().
        merge(combined_accidents_cleaner_w_dst["kazatarihi_full"].dt.year.drop_duplicates().to_frame().rename(columns = {"kazatarihi_full": "year"}), how = "cross").
        merge(combined_accidents_cleaner_w_dst[["two_weeks_after_change"]].drop_duplicates(), how = "cross")
    )


    all_zoomed_in_data_agg = (pd.concat([
        zoomed_in_data_agg, placebo_zoomed_in_data_agg.rename(columns = {"placebo_two_weeks_before_after_change": "two_weeks_before_after_change", 
        "placebo_two_weeks_after_change": "two_weeks_after_change"})
    ]).
    merge(to_balance_data, how = "right", on = ["kaza_ili", "year", "two_weeks_after_change"]).
    assign(**{"post_treatment": lambda x: x["year"].case_when(
        [
            ((x["year"] >= 2017) & (x["two_weeks_after_change"] == True), 1.),
            (pd.Series(True), 0.)
        ]
    ) }).
    assign(**{"pre_treatment": lambda x: x["year"].case_when(
        [
            ((x["year"] < 2017) & (x["two_weeks_after_change"] == True), 1.),
            (pd.Series(True), 0.)
        ]
    ) }).
    assign(**{"total_accidents": lambda x: x["total_accidents"].fillna(0)}).
    assign(**{"real_part": lambda x: x["year"] < 2017}).
    assign(**{"cluster_var": lambda x: x["year"].astype(str) + "-" + x["two_weeks_after_change"].astype(str)}).
    assign(**{"year": lambda x: x["year"].astype(int)}).
    assign(**{"kaza_ili": lambda x: x["kaza_ili"].astype(int)})


    )


