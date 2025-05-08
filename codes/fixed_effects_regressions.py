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


num_weeks_diff = 5
combined_accidents_cleaner_w_dst = (combined_accidents_cleaner.
assign(**{"accident_year_month": lambda x: x["kazatarihi_full"].dt.strftime("%Y-%m")}).
assign(**{"kazatarihi_full2": lambda x: x["kazatarihi_full"]}).
assign(**{"kazatarihi_full": lambda x: x["kazatarihi_full"].dt.floor('D')}).
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
assign(**{"accident_time": lambda x: x["kazatarihi_full2"].dt.time})

)
federal_holidays = pd.read_parquet(f"{home_directory}federal_holidays_turkey_table.parquet").rename(columns = {"date": "kazatarihi_date"})


start_hour = 7
end_hour = 8
start_time = time(start_hour,0,0)
end_time = time(end_hour,0,0)


combined_accidents_cleaner_w_dst_holidays = (combined_accidents_cleaner_w_dst.
assign(**{"kazatarihi_date": lambda x: x["kazatarihi_full"].dt.strftime("%Y-%m-%d")}).
merge(federal_holidays, on = ["kazatarihi_date"], how = "left")
#.query("@start_time <= accident_time <= @end_time")
.assign(**{"holiday": lambda x: x["holiday"].fillna("no_holiday")})
#.query("holiday != holiday")
)




zoomed_in_data_agg = (combined_accidents_cleaner_w_dst_holidays.query("kazatarihi_full.dt.year < 2017").
groupby(["kazatarihi_date", "kaza_ili", "two_weeks_after_change", "holiday"]).agg(total_accidents = ("kazaid", "count")).reset_index()
)

placebo_zoomed_in_data_agg = (combined_accidents_cleaner_w_dst_holidays.query("kazatarihi_full.dt.year >= 2017").
groupby(["kazatarihi_date", "kaza_ili", "placebo_two_weeks_after_change", "holiday"]).agg(total_accidents = ("kazaid", "count")).reset_index()
)

placebo_zoomed_in_data_agg.columns = zoomed_in_data_agg.columns
zoomed_in_data_together = pd.concat([zoomed_in_data_agg, placebo_zoomed_in_data_agg], ignore_index= True)

to_balance_data = (zoomed_in_data_together[["kazatarihi_date", "two_weeks_after_change", "holiday"]].drop_duplicates().
merge(zoomed_in_data_agg[["kaza_ili"]].drop_duplicates(), how = "cross"))

zoomed_in_data_together = (zoomed_in_data_together.
merge(to_balance_data, on = ["kazatarihi_date", "kaza_ili", "two_weeks_after_change"], how = "right").
assign(**{"total_accidents": lambda x: x["total_accidents"].fillna(0)}).
assign(**{"lag_post_pre": lambda x: x.sort_values(["kaza_ili", "kazatarihi_date"]).groupby(["kaza_ili"])["two_weeks_after_change"].shift(1)}).
assign(**{"kazatarihi_nextdate": lambda x: x.sort_values(["kaza_ili", "kazatarihi_date"]).groupby(["kaza_ili"])["kazatarihi_date"].shift(1)}).
assign(**{"start_date": lambda x: x["kazatarihi_date"].case_when(
    [
        ((x["lag_post_pre"] == x["lag_post_pre"]) & (x["lag_post_pre"] != x["two_weeks_after_change"]) & (x["two_weeks_after_change"] == True) , x["kazatarihi_date"] ), 
        (pd.Series(True), pd.NA)
    ]
)}).
assign(**{"kazatarihi_date": lambda x: pd.to_datetime(x["kazatarihi_date"])})
)

list_of_switch_dates = list(zoomed_in_data_together.query("start_date == start_date")["start_date"].unique())
for index, keys in enumerate(list_of_switch_dates):
    print(keys)
    relevant_date = pd.to_datetime(keys)
    zoomed_in_data_together = (zoomed_in_data_together.
        assign(**{f"relative_to_switch_date_{index}": lambda x: (x["kazatarihi_date"] - relevant_date).dt.days  }).
        assign(**{f"abs_relative_to_switch_date_{index}": lambda x: abs(x[f"relative_to_switch_date_{index}"])})
        
        )
        
zoomed_in_data_together['min_abs_relative_diff'] = zoomed_in_data_together.filter(like='abs_relative_to_switch_date_').min(axis=1)

rel_diff_cols = zoomed_in_data_together.filter(like='relative_to_switch_date_')

# 2. Get the column name with minimum absolute value for each row
min_col_idx = rel_diff_cols.abs().idxmin(axis=1)

# 3. Create new column by using numpy array indexing
zoomed_in_data_together['min_relative_diff'] = rel_diff_cols.values[np.arange(len(zoomed_in_data_together)), 
                                              [rel_diff_cols.columns.get_loc(col) for col in min_col_idx]]


to_drop_columns1 = list(zoomed_in_data_together.filter(like = "abs_relative_to_switch_date_").columns)
to_drop_columns2 = list(zoomed_in_data_together.filter(like = "relative_to_switch_date_").columns)
to_drop_columns = to_drop_columns1 + to_drop_columns2 + ["holiday_x"]
zoomed_in_data_together = zoomed_in_data_together.drop(columns = to_drop_columns)

num_days = 28

zoomed_in_data_agg_city = (zoomed_in_data_together.
groupby(["kazatarihi_date", "two_weeks_after_change", "holiday_y", "min_abs_relative_diff", "min_relative_diff"]).
agg(total_accidents = ('total_accidents', 'sum')).reset_index().
assign(**{"accident_month_day": lambda x: x["kazatarihi_date"].dt.strftime("%m-%d")}).
assign(**{"accident_year": lambda x : x["kazatarihi_date"].dt.year}).
assign(**{"treatment_years": lambda x: (x["kazatarihi_date"].dt.year >= 2013) & (x["kazatarihi_date"].dt.year <= 2016)}).
assign(**{"treatment_windows": lambda x: (x["treatment_years"] == True) & (x["min_relative_diff"] >= -num_days) & (x["min_relative_diff"] < num_days)}).
assign(**{"treatment_windows_spring": lambda x: (x["treatment_windows"] == True) & (x["kazatarihi_date"].dt.month <= 6)  }).
assign(**{"treatment_windows_fall": lambda x: (x["treatment_windows"] == True) & (x["kazatarihi_date"].dt.month > 6)  }).
assign(**{"post_treatment_spring": lambda x: (x["treatment_windows_spring"] == True) & (x["min_relative_diff"] >= 0)  }).
assign(**{"post_treatment_fall": lambda x: (x["treatment_windows_fall"] == True) & (x["min_relative_diff"] >= 0)  })

)

fitted_values = pf.fepois("total_accidents ~ post_treatment_spring + post_treatment_fall  | accident_month_day + accident_year + treatment_windows_spring + treatment_windows_fall", data = zoomed_in_data_agg_city.query("holiday_y == 'no_holiday'"))
fitted_values.summary()
