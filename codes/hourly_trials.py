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
assign(**{"accident_time": lambda x: x["kazatarihi_full"].dt.time})

)

federal_holidays = pd.read_parquet(f"{home_directory}federal_holidays_turkey_table.parquet").rename(columns = {"date": "kazatarihi_date"})


start_hour = 6
end_hour = 9
start_time = time(start_hour,0,0)
end_time = time(end_hour,0,0)


combined_accidents_cleaner_w_dst_holidays = (combined_accidents_cleaner_w_dst.
assign(**{"kazatarihi_date": lambda x: x["kazatarihi_full"].dt.strftime("%Y-%m-%d")}).
merge(federal_holidays, on = ["kazatarihi_date"], how = "left")
.query("@start_time <= accident_time <= @end_time")
.assign(**{"holiday": lambda x: x["holiday"].fillna("no_holiday")})
#.query("holiday != holiday")
)




zoomed_in_data_agg = (combined_accidents_cleaner_w_dst_holidays.query("two_weeks_before_after_change == True").
groupby(["kazatarihi_date", "kaza_ili", "two_weeks_after_change", "holiday"]).agg(total_accidents = ("kazaid", "count")).reset_index()
)

placebo_zoomed_in_data_agg = (combined_accidents_cleaner_w_dst_holidays.query("placebo_two_weeks_before_after_change == True").
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

zoomed_in_data_together.info()
list_of_switch_dates = list(zoomed_in_data_together.query("start_date == start_date")["start_date"].unique())
for index, keys in enumerate(list_of_switch_dates):
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

zoomed_in_data_together["min_relative_diff"].unique()

zoomed_in_data_together = (zoomed_in_data_together.
    assign(**{"window": lambda x: ((x["min_relative_diff"] >= -14)  & (x["min_relative_diff"] <= 13)).astype(int) }).
    assign(**{"real_window": lambda x: ((x["min_relative_diff"] >= -14)  & (x["min_relative_diff"] <= 13) & (x["kazatarihi_date"].dt.year < 2017)).astype(int) }).
    assign(**{"pre_treatment": lambda x: (  (x["min_relative_diff"] < 0)  & (x["window"] == 1)  ).astype(int) }).
    assign(**{"day_of_the_week": lambda x: x["kazatarihi_date"].dt.dayofweek}).
    sort_values(["kazatarihi_date", "kaza_ili"]).
    assign(**{"start_indic": lambda x: (x["min_relative_diff"] == x["min_relative_diff"].min()) & (x["kaza_ili"] == x["kaza_ili"].min())}).
    assign(**{"experiment_num": lambda x: x["start_indic"].cumsum()})
    )

for index,key in enumerate(list(zoomed_in_data_together["experiment_num"].unique())):
    zoomed_in_data_together = (zoomed_in_data_together.
        assign(**{f"trend_{index}": lambda x: (x["experiment_num"] == key) * x["min_relative_diff"]})
        )



final_reg_data = zoomed_in_data_together.copy()

for zz in range(-3, 3): 
    if zz < 0:
        initial = "m"
    else: 
        initial = "p"
    zz_name = str(abs(zz))
    final_reg_data = (final_reg_data.
        assign(**{f"treatment_{initial}_{zz_name}": lambda x: (  (x["min_relative_diff"] >= zz * 7) & (x["min_relative_diff"] < (zz+1) * 7)  ).astype(int) }).
        assign(**{f"real_treatment_{initial}_{zz_name}": lambda x: (  (x["min_relative_diff"] >= zz * 7) & (x["min_relative_diff"] < (zz+1) * 7) & (x["kazatarihi_date"].dt.year < 2017)).astype(int) })
        )



treatment_vars = [col for col in final_reg_data.columns if col.startswith("treatment_")]
treatment_vars.pop(2)
real_treatment_vars = [col for col in final_reg_data.columns if col.startswith("real_treatment_")]
real_treatment_vars.pop(np.argwhere(np.array(real_treatment_vars) ==  "real_treatment_m_1")[0][0])

trend_vars = [col for col in final_reg_data.columns if col.startswith("trend_")]
treatment_vars_in_formula = " + ".join(treatment_vars)
real_treatment_vars_in_formula = " + ".join(real_treatment_vars)
trend_vars_in_formula = " + ".join(trend_vars)

results = pf.feols(f"total_accidents ~  {real_treatment_vars_in_formula} + min_relative_diff | kaza_ili + day_of_the_week + real_window + {treatment_vars_in_formula}  ", 
data = final_reg_data.query("holiday_y == 'no_holiday'"))
results.summary()




final_reg_data.shape
final_reg_data.shape[0]/13
