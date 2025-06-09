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
assign(**{"accident_time": lambda x: x["kazatarihi_full"].dt.time})

)

