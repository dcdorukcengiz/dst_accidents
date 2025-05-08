%reset -f

import numpy as np
import pandas as pd
import janitor
import matplotlib.pyplot as plt
import matplotlib as mp
import pyfixest as pf
from datetime import time, datetime
from statsmodels.nonparametric.smoothers_lowess import lowess
from rdrobust import rdrobust,rdbwselect,rdplot
import matplotlib.ticker as ticker  


home_directory = "/Users/EzgilovesDoruk/Desktop/education_health/"
combined_accidents_cleaner = pd.read_parquet(f"{home_directory}combined_accidents_cleaner_table.parquet")

#Monthly Accidents

monthly_totals = (combined_accidents_cleaner.
    assign(**{"month": lambda x: x["kazatarihi_full"].dt.month}).
    assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).
    groupby(["month", "year"]).
    agg(total_accidents = ('kazaid', 'count')).
    reset_index()
    )


# Define colors for each year (using a colormap)
years = monthly_totals['year'].unique() # Get unique years
colors = plt.cm.get_cmap('viridis', len(years))

# Create the plot
plt.figure(figsize=(10, 6))

for i, year in enumerate(years):
    year_data = monthly_totals[monthly_totals['year'] == year] # Filter data for current year
    plt.plot(year_data['month'], year_data['total_accidents'], marker='o', linestyle='-', color=colors(i), label=str(year))


# Customize the plot
plt.xlabel('Month', fontsize=12)
plt.ylabel('# Accidents', fontsize=12)
plt.title('Monthly Total Accidents by Year', fontsize=14)
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.ylim(bottom=0)

# Show the plot
plt.show()

num_weeks_diff = 4
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

relative_to_dst_change = (
    combined_accidents_cleaner.
    assign(**{"relative_to_dst_2013": lambda x: np.floor((x["kazatarihi_full"] - pd.to_datetime('2013-03-31'))/pd.Timedelta(days = 1))}).
    assign(**{"relative_to_dst_2014": lambda x: np.floor((x["kazatarihi_full"] - pd.to_datetime('2014-03-30'))/pd.Timedelta(days = 1))}).
    assign(**{"relative_to_dst_2015": lambda x: np.floor((x["kazatarihi_full"] - pd.to_datetime('2015-03-29'))/pd.Timedelta(days = 1))}).
    assign(**{"relative_to_dst_2016": lambda x: np.floor((x["kazatarihi_full"] - pd.to_datetime('2016-03-27'))/pd.Timedelta(days = 1))}).
    assign(**{"relative_to_dst_2017": lambda x: np.floor((x["kazatarihi_full"] - pd.to_datetime('2017-04-02'))/pd.Timedelta(days = 1))}).
    assign(**{"relative_to_dst_2018": lambda x: np.floor((x["kazatarihi_full"] - pd.to_datetime('2018-04-01'))/pd.Timedelta(days = 1))}).
    assign(**{"relative_to_dst_2019": lambda x: np.floor((x["kazatarihi_full"] - pd.to_datetime('2019-03-31'))/pd.Timedelta(days = 1))})

)

relative_to_st_change = (
    combined_accidents_cleaner_w_dst.
    assign(**{"relative_to_dst_2013": lambda x: np.floor((x["kazatarihi_full"] - pd.to_datetime('2013-10-27'))/pd.Timedelta(days = 1))}).
    assign(**{"relative_to_dst_2014": lambda x: np.floor((x["kazatarihi_full"] - pd.to_datetime('2014-10-26'))/pd.Timedelta(days = 1))}).
    assign(**{"relative_to_dst_2015": lambda x: np.floor((x["kazatarihi_full"] - pd.to_datetime('2015-11-08'))/pd.Timedelta(days = 1))}).
    assign(**{"relative_to_dst_2017": lambda x: np.floor((x["kazatarihi_full"] - pd.to_datetime('2017-10-29'))/pd.Timedelta(days = 1))}).
    assign(**{"relative_to_dst_2018": lambda x: np.floor((x["kazatarihi_full"] - pd.to_datetime('2018-10-28'))/pd.Timedelta(days = 1))}).
    assign(**{"relative_to_dst_2016": lambda x: np.floor((x["kazatarihi_full"] - pd.to_datetime('2016-10-30'))/pd.Timedelta(days = 1))})
 
)



def cleaner_func(data_itself, year, window_size):
    start_date = f"{year}-01-01"
    end_date = f"{year + 1}-01-01"
    temp =  (data_itself.query("kazatarihi_full >= @start_date and kazatarihi_full <= @end_date").
    query(f"-@window_size <= relative_to_dst_{year} <= (@window_size + 6)").
    assign(**{"relative_to_real_dst": lambda x: x[f"relative_to_dst_{year}"]})
    )

    cols_to_drop = [col for col in temp.columns if col.startswith('relative_to_dst_')]
    return temp.drop(columns = cols_to_drop)
   



func_window_size = 42
relative_to_dst_dict = {}
for zzz in range(2013, 2020):
    print(zzz)
    relative_to_dst_dict[zzz] = cleaner_func(relative_to_dst_change, zzz, func_window_size).sort_values(["kazaid"])

all_relative_to_dst = (pd.concat(relative_to_dst_dict, ignore_index = True).reset_index(drop = True).
assign(**{"placebo_years": lambda x: x["kazatarihi_full"] >= '2016-06-30'}).
assign(**{"year": lambda x: x["kazatarihi_full"].dt.year})
)

relative_to_st_dict = {}
for zzz in [2013,2014,2015, 2016, 2017, 2018]:
    print(zzz)
    relative_to_st_dict[zzz] = cleaner_func(relative_to_st_change, zzz, func_window_size).sort_values(["kazaid"])

all_relative_to_st = (pd.concat(relative_to_st_dict, ignore_index = True).reset_index(drop = True).
assign(**{"placebo_years": lambda x: x["kazatarihi_full"] >= '2016-06-30'}).
assign(**{"year": lambda x: x["kazatarihi_full"].dt.year})
)
temptemp = all_relative_to_st.query("year == 2016")[["kazatarihi"]].drop_duplicates()


for_figure_all_years_dst = all_relative_to_dst.groupby(["relative_to_real_dst", "placebo_years", "year"]).agg(total_accidents = ('kazaid', 'count')).reset_index()
for_figure_all_years_st = all_relative_to_st.groupby(["relative_to_real_dst", "placebo_years", "year"]).agg(total_accidents = ('kazaid', 'count')).reset_index()
for_figure_all_years = pd.concat([for_figure_all_years_dst, for_figure_all_years_st], ignore_index = True).groupby(["relative_to_real_dst", "placebo_years", "year"]).agg(total_accidents = ('total_accidents', 'sum')).reset_index()

for_figure_all_years = (for_figure_all_years.
    assign(**{"placebo": lambda x: x["placebo_years"].astype(str)}).
    assign(**{"year_str": lambda x: x["year"].astype(str)}).    
    assign(**{"placebo_year": lambda x:  x["placebo_years"].astype(str) + "_" + x["year"].astype(str)})
    )

for zz in for_figure_all_years["placebo_years"].unique():

    placebo_indic = zz

    for_figure_all_years = (for_figure_all_years.
        assign(**{f"spec_trend_{zz}": lambda x: x["relative_to_real_dst"].case_when(
            [
                (((x["placebo_years"] == placebo_indic)), x["relative_to_real_dst"] ), 
                (pd.Series(True) , 0)
            ]
        )}))

for_figure_all_years = (for_figure_all_years.
    assign(**{"weeks": lambda x: np.floor(x["relative_to_real_dst"]/7)}).
    assign(**{"placebo_strings": lambda x: x["placebo_years"].astype(str)}).
    assign(**{"post_treatment_m3": lambda x: (x["relative_to_real_dst"] < 0) & (x["placebo_years"] == False) & (x["weeks"] == -3)}).
    assign(**{"post_treatment_m2": lambda x: (x["relative_to_real_dst"] < 0) & (x["placebo_years"] == False) & (x["weeks"] == -2)}).
    assign(**{"post_treatment_m1": lambda x: (x["relative_to_real_dst"] < 0) & (x["placebo_years"] == False) & (x["weeks"] == -1)}).
    assign(**{"post_treatment_p0": lambda x: (x["relative_to_real_dst"] >= 0) & (x["placebo_years"] == False) & (x["weeks"] == 0)}).
    assign(**{"post_treatment_p1": lambda x: (x["relative_to_real_dst"] >= 0) & (x["placebo_years"] == False) & (x["weeks"] == 1)}).
    assign(**{"post_treatment_p2": lambda x: (x["relative_to_real_dst"] >= 0) & (x["placebo_years"] == False) & (x["weeks"] == 2)}).
    assign(**{"post_treatment_p3": lambda x: (x["relative_to_real_dst"] >= 0) & (x["placebo_years"] == False) & (x["weeks"] == 3)}).
    assign(**{"post_treatment_p4": lambda x: (x["relative_to_real_dst"] >= 0) & (x["placebo_years"] == False) & (x["weeks"] == 4)}).
    assign(**{"treatment": lambda x: x["relative_to_real_dst"] >= 0 })

)

spec_trend_cols = [col for col in for_figure_all_years.columns if col.startswith('spec_trend_')]

pf.feols(f"""total_accidents ~ post_treatment_m2+post_treatment_p0+post_treatment_p1+post_treatment_p2 + post_treatment_p3  | relative_to_real_dst +  placebo_strings""", data = for_figure_all_years, vcov = "HC1").summary()




a = f"""total_accidents ~ post_treatment_m3+post_treatment_m2+post_treatment_p0+post_treatment_p1+post_treatment_p2 +post_treatment_p3 + {" + ".join(spec_trend_cols)} | relative_to_real_dst + placebo_strings"""

a