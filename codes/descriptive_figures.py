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
import matplotlib.dates as mdates


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



### Sunrise and sunset hours

#When there is DST-ST change 
df = pd.read_parquet(f"{home_directory}city_sunrise_sunset_table.parquet")
df = (df.
assign(**{"sunrise_date": lambda x: x["date"].astype(str) + "-" + x["sunrise"]}).
assign(**{"sunset_date": lambda x: x["date"].astype(str) + "-" + x["sunset"]}).
assign(**{"sunrise_date": lambda x: pd.to_datetime(x["sunrise_date"])}).
assign(**{"sunset_date": lambda x: pd.to_datetime(x["sunset_date"])}).
query("kaza_ili == 6").
query("sunrise_date.dt.year == 2013 or sunrise_date.dt.year == 2018").
assign(**{"sunrise_time": lambda x: x["sunrise_date"].dt.time}).
assign(**{"sunset_time": lambda x: x["sunset_date"].dt.time})
)



# 2. Plotting

# *** KEY CORRECTION: Convert time to datetime using a dummy date ***
dummy_date = df['sunrise_date'].iloc[0].date()  # Use any date from your data as the dummy
df['sunrise_time_datetime'] = df['sunrise_time'].apply(lambda x: datetime.combine(dummy_date, x) if x else None)
df['sunset_time_datetime'] = df['sunset_time'].apply(lambda x: datetime.combine(dummy_date, x) if x else None)


plt.figure(figsize=(10, 6))
plt.plot(df.query("sunrise_date.dt.year == 2013")['sunrise_date'].dt.date, df.query("sunrise_date.dt.year == 2013")['sunrise_time_datetime'], marker='o', alpha = 0.2, linestyle='-', markersize=2, color = "orange")
plt.plot(df.query("sunrise_date.dt.year == 2013")['sunrise_date'].dt.date, df.query("sunrise_date.dt.year == 2013")['sunset_time_datetime'], marker='o', alpha = 0.2, linestyle='-', markersize=2, color = "orange")
plt.plot(df.query("sunrise_date.dt.year == 2013")['sunrise_date'].dt.date, df.query("sunrise_date.dt.year == 2018")['sunrise_time_datetime'], marker='o',  markersize= 2, alpha = 0.2, color = "blue")
plt.plot(df.query("sunrise_date.dt.year == 2013")['sunrise_date'].dt.date, df.query("sunrise_date.dt.year == 2018")['sunset_time_datetime'], marker='o',  markersize= 2, alpha = 0.2, color = "blue")
plt.fill_between(df.query("sunrise_date.dt.year == 2013")['sunrise_date'].dt.date, df.query("sunrise_date.dt.year == 2018")['sunset_time_datetime'], df.query("sunrise_date.dt.year == 2013")['sunset_time_datetime'], alpha = 0.5, color = "orange")
plt.fill_between(df.query("sunrise_date.dt.year == 2013")['sunrise_date'].dt.date, df.query("sunrise_date.dt.year == 2018")['sunrise_time_datetime'], df.query("sunrise_date.dt.year == 2013")['sunrise_time_datetime'], alpha = 0.5, color = "orange")

plt.xlabel('Date')
plt.ylabel('Time (HH:MM)')
plt.title('')
plt.grid(True)

# Format x-axis (dates)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
plt.gcf().autofmt_xdate()
plt.grid(True, alpha=0.3, linewidth=0.5)
# Format y-axis (times)
plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

plt.tight_layout()





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
   



func_window_size = 49
#14 works. Need to find others.
relative_to_dst_dict = {}
for zzz in range(2013, 2020):
    print(zzz)
    relative_to_dst_dict[zzz] = cleaner_func(relative_to_dst_change, zzz, func_window_size).sort_values(["kazaid"])

all_relative_to_dst = (pd.concat(relative_to_dst_dict, ignore_index = True).reset_index(drop = True).
assign(**{"placebo_years": lambda x: x["kazatarihi_full"] >= '2016-07-15'})
)


relative_to_st_dict = {}
for zzz in [2013,2014,2015, 2016, 2017, 2018]:
    print(zzz)
    relative_to_st_dict[zzz] = cleaner_func(relative_to_st_change, zzz, func_window_size).sort_values(["kazaid"])

all_relative_to_st = (pd.concat(relative_to_st_dict, ignore_index = True).reset_index(drop = True).
assign(**{"placebo_years": lambda x: x["kazatarihi_full"] >= '2016-06-30'}))

all_relative_to_dst.to_parquet(f"{home_directory}all_relative_to_dst.parquet")
all_relative_to_st.to_parquet(f"{home_directory}all_relative_to_st.parquet")

for_figure_all_years_dst = all_relative_to_dst.groupby(["relative_to_real_dst", "placebo_years"]).agg(total_accidents = ('kazaid', 'count')).reset_index()
for_figure_all_years_st = all_relative_to_st.groupby(["relative_to_real_dst", "placebo_years"]).agg(total_accidents = ('kazaid', 'count')).reset_index()

for_figure_all_years = (pd.concat([for_figure_all_years_dst, for_figure_all_years_st], ignore_index=True).
groupby(["relative_to_real_dst", "placebo_years"]).agg(total_accidents = ('total_accidents', 'mean')).reset_index())


multiplier_dst = (for_figure_all_years_dst.query("placebo_years == False and relative_to_real_dst < 0")["total_accidents"].mean())/(for_figure_all_years_dst.query("placebo_years == True and relative_to_real_dst < 0")["total_accidents"].mean())
multiplier_st = (for_figure_all_years_st.query("placebo_years == False and relative_to_real_dst < 0")["total_accidents"].mean())/(for_figure_all_years_st.query("placebo_years == True and relative_to_real_dst < 0")["total_accidents"].mean())
multiplier = (for_figure_all_years.query("placebo_years == False and relative_to_real_dst < 0")["total_accidents"].mean())/(for_figure_all_years.query("placebo_years == True and relative_to_real_dst < 0")["total_accidents"].mean())


for_figure_all_years_dst = (for_figure_all_years_dst.
    assign(**{"total_accidents": lambda x: x["total_accidents"].
    case_when(
        [
            (x["placebo_years"] == True , x["total_accidents"] * multiplier_dst),
            (pd.Series(True), x["total_accidents"])
        ]
    )
    }))

for_figure_all_years = (for_figure_all_years.
    assign(**{"total_accidents": lambda x: x["total_accidents"].
    case_when(
        [
            (x["placebo_years"] == True , x["total_accidents"] * multiplier),
            (pd.Series(True), x["total_accidents"])
        ]
    )
    }))


for_figure_all_years_st = (for_figure_all_years_st.
    assign(**{"total_accidents": lambda x: x["total_accidents"].
    case_when(
        [
            (x["placebo_years"] == True , x["total_accidents"] * multiplier_st),
            (pd.Series(True), x["total_accidents"])
        ]
    )
    }))


colors = ['orange' if val == True else 'blue' for val in for_figure_all_years_dst['placebo_years']]

plt.figure(figsize=(8, 6))
plt.scatter(for_figure_all_years_dst['relative_to_real_dst'], for_figure_all_years_dst['total_accidents'], c=colors, marker='o', label='Data Points')  # Use 'c' for color

colors = ['orange' if val == True else 'blue' for val in for_figure_all_years['placebo_years']]

plt.figure(figsize=(8, 6))
plt.scatter(for_figure_all_years['relative_to_real_dst'], for_figure_all_years['total_accidents'], c=colors, marker='o', label='Data Points')  # Use 'c' for color

colors = ['orange' if val == True else 'blue' for val in for_figure_all_years_st['placebo_years']]

plt.figure(figsize=(8, 6))
plt.scatter(for_figure_all_years_st['relative_to_real_dst'], for_figure_all_years_st['total_accidents'], c=colors, marker='o', label='Data Points')  # Use 'c' for color


temp_data = (for_figure_all_years.
assign(**{"treatment": lambda x: x["relative_to_real_dst"] >= 0 }).
assign(**{"post_treatment": lambda x: (x["relative_to_real_dst"] >= 0) & (x["placebo_years"] == False)})
)

for_figure_all_years.columns
smoothed1 = lowess(exog=for_figure_all_years.query("placebo_years == True")['relative_to_real_dst'], 
        endog=for_figure_all_years.query("placebo_years == True")['total_accidents'],
        frac = 0.4)

smoothed2 = lowess(exog=for_figure_all_years.query("placebo_years == False")['relative_to_real_dst'], 
        endog=for_figure_all_years.query("placebo_years == False")['total_accidents'],
        frac = 0.4)


fig, ax = plt.subplots()
ax.plot(smoothed1[:, 0], smoothed1[:, 1], c='orange')
ax.plot(smoothed2[:, 0], smoothed2[:, 1], c='blue')


weekly_reg = (for_figure_all_years.
    assign(**{"weeks": lambda x: np.floor(x["relative_to_real_dst"]/7)}).
    assign(**{"placebo_strings": lambda x: x["placebo_years"].astype(str)}).
    assign(**{"post_treatment_m4": lambda x: (x["relative_to_real_dst"] < 0) & (x["placebo_years"] == False) & (x["weeks"] <= -4)}).
    assign(**{"post_treatment_m3": lambda x: (x["relative_to_real_dst"] < 0) & (x["placebo_years"] == False) & (x["weeks"] == -3)}).
    assign(**{"post_treatment_m2": lambda x: (x["relative_to_real_dst"] < 0) & (x["placebo_years"] == False) & (x["weeks"] == -2)}).
    assign(**{"post_treatment_m1": lambda x: (x["relative_to_real_dst"] < 0) & (x["placebo_years"] == False) & (x["weeks"] == -1)}).
    assign(**{"post_treatment_p0": lambda x: (x["relative_to_real_dst"] >= 0) & (x["placebo_years"] == False) & (x["weeks"] == 0)}).
    assign(**{"post_treatment_p1": lambda x: (x["relative_to_real_dst"] >= 0) & (x["placebo_years"] == False) & (x["weeks"] == 1)}).
    assign(**{"post_treatment_p2": lambda x: (x["relative_to_real_dst"] >= 0) & (x["placebo_years"] == False) & (x["weeks"] == 2)}).
    assign(**{"post_treatment_p3": lambda x: (x["relative_to_real_dst"] >= 0) & (x["placebo_years"] == False) & (x["weeks"] == 3)}).
    assign(**{"post_treatment_p4": lambda x: (x["relative_to_real_dst"] >= 0) & (x["placebo_years"] == False) & (x["weeks"] == 4)}).
    assign(**{"post_treatment_p5": lambda x: (x["relative_to_real_dst"] >= 0) & (x["placebo_years"] == False) & (x["weeks"] == 5)}).
    assign(**{"post_treatment_p6": lambda x: (x["relative_to_real_dst"] >= 0) & (x["placebo_years"] == False) & (x["weeks"] >= 6)}).
    assign(**{"relative_to_real_dst": lambda x: x["relative_to_real_dst"].astype(float)}).
    assign(**{"treatment": lambda x: x["relative_to_real_dst"] >= 0 }).
    assign(**{"post_treatment": lambda x: (x["relative_to_real_dst"] >= 0) & (x["placebo_years"] == False)})

)

weekly_reg.to_parquet(f"{home_directory}weekly_reg_table.parquet")


pf.feols("""total_accidents ~ post_treatment_m3+post_treatment_m2+post_treatment_p0+post_treatment_p1+post_treatment_p2 | relative_to_real_dst + placebo_strings""", data = weekly_reg, vcov = "HC1").summary()
pf.fepois("""total_accidents ~ post_treatment_m4+post_treatment_m3+post_treatment_m2+post_treatment_p0+post_treatment_p1+post_treatment_p2 + post_treatment_p3 + post_treatment_p4 +  post_treatment_p5 +  post_treatment_p6 + relative_to_real_dst | weeks + placebo_strings""", data = weekly_reg, vcov = "HC1").summary()
pf.feols("""total_accidents ~ post_treatment_m4+post_treatment_m3+post_treatment_m2+post_treatment_p0+post_treatment_p1+post_treatment_p2 + post_treatment_p3 + post_treatment_p4 +  post_treatment_p5 +  post_treatment_p6  | weeks + placebo_strings""", data = weekly_reg, vcov = "HC1").summary()

fit_results = pf.fepois("""total_accidents ~ post_treatment | treatment + placebo_strings """, vcov = "HC1", data = weekly_reg)
fit_results.summary()


fit_results = pf.fepois("""total_accidents ~ post_treatment_m3+post_treatment_m2+post_treatment_p0+post_treatment_p1+post_treatment_p2 + post_treatment_p3 + post_treatment_p4 + relative_to_real_dst | weeks + placebo_strings""", data = weekly_reg, vcov = "HC1")
np.sqrt(fit_results._vcov[0,0])
fit_results.summary()
dir(fit_results)
coefs = fit_results.coef().reset_index()
ses = fit_results.se().reset_index()


temp = (weekly_reg.copy().
assign(**{"temptemp": lambda x: x["post_treatment_m3"] + x["post_treatment_m2"] + x["post_treatment_m1"] + x["post_treatment_p0"] + x["post_treatment_p1"] + x["post_treatment_p2"] + x["post_treatment_p3"] })
)
temp["temptemp"].unique()
asd = temp.query("placebo_years == False and temptemp == False")

error_bar_plot = (coefs.merge(ses, on = ["Coefficient"]).assign(**{"relative_to_event": lambda x: [-2,0,1,2]}))
reference_date = pd.DataFrame({"A": [0],
"B": [0],
"C": [0], 
"D": [-1] })
reference_date.columns = error_bar_plot.columns
error_bar_plot = pd.concat([error_bar_plot, reference_date], ignore_index = True).sort_values(["relative_to_event"])


plt.figure(figsize=(10, 6))  
plt.errorbar(error_bar_plot['relative_to_event'], error_bar_plot['Estimate'], yerr=error_bar_plot['Std. Error'], 
             fmt='o-',  # 'o-' creates markers and connects them with lines
             capsize=5,  # adds caps to the error bars
             label='Coefficients with Error Bars',
             color = 'blue')
plt.xticks(error_bar_plot['relative_to_event'], error_bar_plot['relative_to_event'])
plt.axvline(x=0, color='black', linestyle='--', label='Zero Line')  # Add the vertical line
formatter = ticker.PercentFormatter(xmax=1, decimals=0, symbol='%') # xmax=1.0 for data between 0 and 1
plt.gca().yaxis.set_major_formatter(formatter)



pf.feols("total_accidents ~ treatment + post_treatment", data = temp_data).summary()
pf.fepois("total_accidents ~ treatment + post_treatment", data = temp_data).summary()

pf.feols("total_accidents ~ 1", data = temp_data.query("placebo_years == True and relative_to_real_dst < 0")).summary()
pf.feols("total_accidents ~ 1", data = temp_data.query("placebo_years == True and relative_to_real_dst >= 0")).summary()

pf.feols("total_accidents ~ 1", data = temp_data.query("placebo_years == False and relative_to_real_dst < 0")).summary()
pf.feols("total_accidents ~ 1", data = temp_data.query("placebo_years == False and relative_to_real_dst >= 0")).summary()


for_rdd = all_relative_to_dst.assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).groupby(["relative_to_real_dst", "year", "placebo_years"]).agg(total_accidents = ('kazaid', 'count')).reset_index()
for_rdd_1 = for_rdd.query("placebo_years == True")
print(rdrobust(y=for_rdd_1["total_accidents"], x=for_rdd_1["relative_to_real_dst"]))

rdplot(y=for_rdd_1["total_accidents"], x=for_rdd_1["relative_to_real_dst"], binselect="es" )

all_relative_to_dst.columns

temp_data_dst = (for_figure_all_years_dst.
assign(**{"treatment": lambda x: x["relative_to_real_dst"] >= 0 }).
assign(**{"post_treatment": lambda x: (x["relative_to_real_dst"] >= 0) & (x["placebo_years"] == False)})
)


pf.feols("total_accidents ~ treatment + post_treatment", data = temp_data_dst).summary()


temp_data_st = (for_figure_all_years_st.
assign(**{"treatment": lambda x: x["relative_to_real_dst"] >= 0 }).
assign(**{"post_treatment": lambda x: (x["relative_to_real_dst"] >= 0) & (x["placebo_years"] == False)})
)


pf.feols("total_accidents ~ treatment + post_treatment", data = temp_data_st).summary()



temp_data = (for_figure_all_years_dst.query("placebo_years == True").
assign(**{"treatment": lambda x: x["relative_to_real_dst"] >= 0 })
)
pf.feols("total_accidents ~ treatment", data = temp_data).summary()


temp_data2 = (for_figure_all_years_dst.query("placebo_years == False").
assign(**{"treatment": lambda x: x["relative_to_real_dst"] >= 0 })
)

pf.feols("total_accidents ~ treatment", data = temp_data2).summary()




temp_data = (for_figure_all_years_st.query("placebo_years == True").
assign(**{"treatment": lambda x: x["relative_to_real_dst"] >= 0 })
)


pf.feols("log(total_accidents) ~ treatment", data = temp_data).summary()


temp_data2 = (for_figure_all_years_st.query("placebo_years == False").
assign(**{"treatment": lambda x: x["relative_to_real_dst"] >= 0 })
)

pf.feols("log(total_accidents) ~ treatment", data = temp_data2).summary()

