%reset -f

import numpy as np
import pandas as pd
import janitor
import matplotlib.pyplot as plt
import pyfixest as pf
from datetime import time


home_directory = "/Users/EzgilovesDoruk/Desktop/education_health/"
combined_accidents_cleaner = pd.read_parquet(f"{home_directory}combined_accidents_cleaner_table.parquet")


total_car_accidents_per_year = (combined_accidents_cleaner.groupby(["kazayili"]).agg(total_accidents_per_year = ('kazaid', 'count')).reset_index())

plt.plot(total_car_accidents_per_year["kazayili"], total_car_accidents_per_year["total_accidents_per_year"], color = "orange")
plt.ylim(ymin=100_000)  # Sets the minimum y-value to 0

num_weeks_diff = 3
combined_accidents_cleaner_w_dst = (combined_accidents_cleaner.
assign(**{"summer_hours": lambda x: ((x["kazatarihi_full"] >= '2013-03-31') & (x["kazatarihi_full"] <= '2013-10-27')) | 
        ((x["kazatarihi_full"] >= '2014-03-30') & (x["kazatarihi_full"] <= '2014-10-26')) | 
        ((x["kazatarihi_full"] >= '2015-03-29') & (x["kazatarihi_full"] <= '2015-11-08')) | 
        ((x["kazatarihi_full"] >= '2016-03-27'))}
       
).
assign(**{"accident_year_month": lambda x: x["kazatarihi_full"].dt.strftime("%Y-%m")}).
assign(**{"two_weeks_before_after_change": lambda x: 
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2013-10-27') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2013-10-27'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2014-10-26') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2014-10-26'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2015-11-08') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2015-11-08'))  
    
    }).
assign(**{"two_weeks_after_change": lambda x: 
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2013-10-27') & (x["kazatarihi_full"] >= '2013-10-27'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2014-10-26') & (x["kazatarihi_full"] >= '2014-10-26'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2015-11-08') & (x["kazatarihi_full"] >= '2015-11-08'))  
    
    }).
assign(**{"placebo_two_weeks_before_after_change": lambda x: 
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2017-10-27') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2017-10-27'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2018-10-26') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2018-10-26'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2019-11-08') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2019-11-08'))      
    }).
assign(**{"placebo_two_weeks_after_change": lambda x: 
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2017-10-27') & (x["kazatarihi_full"] >= '2017-10-27'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2018-10-26') & (x["kazatarihi_full"] >= '2018-10-26'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2019-11-08') & (x["kazatarihi_full"] >= '2019-11-08'))  
    
    }).
assign(**{"accident_time": lambda x: x["kazatarihi_full"].dt.time})

)

start_hour = 7
end_hour = 8

start_time = time(start_hour,0,0)
end_time = time(end_hour,0,0)
temptemp = combined_accidents_cleaner_w_dst.query("two_weeks_before_after_change == True")
############         All Accidents        #######################
zoomed_in_data_agg = (combined_accidents_cleaner_w_dst.query("two_weeks_before_after_change == True")
#.query("accident_time <= @end_time and accident_time >= @start_time")
.assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).groupby(["year", "kaza_ili", "two_weeks_after_change"]).
agg(total_accidents = ('kazaid', 'count')).reset_index())

#pf.("total_accidents ~ two_weeks_after_change", data = zoomed_in_data_agg).summary()

placebo_zoomed_in_data_agg = (combined_accidents_cleaner_w_dst.query("placebo_two_weeks_before_after_change == True").
#query("accident_time <= @end_time and accident_time >= @start_time"). 
assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).groupby(["year", "kaza_ili", "placebo_two_weeks_after_change"]).
agg(total_accidents = ('kazaid', 'count')).reset_index())

#pf.("total_accidents ~ placebo_two_weeks_after_change", data = placebo_zoomed_in_data_agg).summary()

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



fit = pf.feols("total_accidents ~ two_weeks_after_change + post_treatment | kaza_ili + year", data = all_zoomed_in_data_agg, vcov={"CRV1": "kaza_ili"})
fit.summary()

fit = pf.feols("np.log1p(total_accidents) ~ two_weeks_after_change + post_treatment | kaza_ili + year", data = all_zoomed_in_data_agg, vcov={"CRV1": "kaza_ili"})
fit.summary()

fit = pf.fepois("total_accidents ~ two_weeks_after_change + post_treatment | kaza_ili + year", data = all_zoomed_in_data_agg, vcov={"CRV1": "kaza_ili"})
fit.summary()
