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



#start_hour = 7
#end_hour = 8


hour_range = 1
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
    zoomed_in_data_agg = (combined_accidents_cleaner_w_dst.query("two_weeks_before_after_change == True")
    .query("accident_time <= @end_time and accident_time >= @start_time")
    .assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).groupby(["year", "kaza_ili", "two_weeks_after_change"]).
    agg(total_accidents = ('kazaid', 'count')).reset_index())

    placebo_zoomed_in_data_agg = (combined_accidents_cleaner_w_dst.query("placebo_two_weeks_before_after_change == True").
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



#    fit = pf.fepois("total_accidents ~ two_weeks_after_change + pre_treatment | kaza_ili + year", data = all_zoomed_in_data_agg, vcov={"CRV1": "kaza_ili"})
    fit = pf.feols("total_accidents ~ two_weeks_after_change + post_treatment | kaza_ili + year", data = all_zoomed_in_data_agg, vcov={"CRV1": "kaza_ili"})
    fit.summary()
    temp = pd.DataFrame({
        "treat_coefficient": [-float(fit.coef()["post_treatment"])],
        "treat_se": [float(fit.se()["post_treatment"])],
        "treat_pvalue": [float(fit.pvalue()["post_treatment"])],
        "two_weeks_coefficient": [float(fit.coef()["two_weeks_after_change"])],
        "two_weeks_se": [float(fit.se()["two_weeks_after_change"])],
        "two_weeks_pvalue": [float(fit.pvalue()["two_weeks_after_change"])], 
        "start_time": [start_hour], 
        "end_time": [end_hour]

    }
    ).assign(**{"significant": lambda x: x["treat_pvalue"] < 0.05})
    all_estimates = pd.concat([all_estimates, temp], ignore_index= True)



errors = 2 * all_estimates['treat_se']


plt.figure(figsize=(15, 6))  # Adjust figure size for better readability if needed
plt.bar(all_estimates['start_time'], all_estimates['treat_coefficient'] * 100, yerr=errors * 100, capsize=3)

# Add labels and title
plt.xlabel('Hour')
plt.ylabel('Percentage Increase in Accidents')
plt.title('')

plt.xticks(all_estimates['start_time']) # Ensure all x-axis values are shown
plt.grid(axis='y', alpha=0.5) # Add a subtle grid for better readability
#plt.ylim([0, 1])  # Set y-axis limits if needed (e.g. 0 to 1)

plt.tight_layout() # Adjust layout to prevent labels from overlapping
ax = plt.gca()  # Get the current axis
#ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))

plt.savefig(f"{home_directory}impact_of_changing_time_dst_st_first_{num_weeks_diff}_weeks_feols_{hour_range}_hours.png", dpi = 300)


#########      DST separate      ################



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
    zoomed_in_data_agg = (combined_accidents_cleaner_w_dst.query("two_weeks_before_after_change == True and kazatarihi_full.dt.month < 6")
    .query("accident_time <= @end_time and accident_time >= @start_time")
    .assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).groupby(["year", "kaza_ili", "two_weeks_after_change"]).
    agg(total_accidents = ('kazaid', 'count')).reset_index())

    placebo_zoomed_in_data_agg = (combined_accidents_cleaner_w_dst.query("placebo_two_weeks_before_after_change == True and kazatarihi_full.dt.month < 6").
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



#    fit = pf.fepois("total_accidents ~ two_weeks_after_change + pre_treatment | kaza_ili + year", data = all_zoomed_in_data_agg, vcov={"CRV1": "kaza_ili"})
    fit = pf.feols("total_accidents ~ two_weeks_after_change + post_treatment | kaza_ili + year", data = all_zoomed_in_data_agg, vcov={"CRV1": "kaza_ili"})
    fit.summary()
    temp = pd.DataFrame({
        "treat_coefficient": [-float(fit.coef()["post_treatment"])],
        "treat_se": [float(fit.se()["post_treatment"])],
        "treat_pvalue": [float(fit.pvalue()["post_treatment"])],
        "two_weeks_coefficient": [float(fit.coef()["two_weeks_after_change"])],
        "two_weeks_se": [float(fit.se()["two_weeks_after_change"])],
        "two_weeks_pvalue": [float(fit.pvalue()["two_weeks_after_change"])], 
        "start_time": [start_hour], 
        "end_time": [end_hour]

    }
    ).assign(**{"significant": lambda x: x["treat_pvalue"] < 0.05})
    all_estimates = pd.concat([all_estimates, temp], ignore_index= True)



errors = 2 * all_estimates['treat_se']


plt.figure(figsize=(15, 6))  # Adjust figure size for better readability if needed
plt.bar(all_estimates['start_time'], all_estimates['treat_coefficient'] * 100, yerr=errors * 100, capsize=3)

# Add labels and title
plt.xlabel('Hour')
plt.ylabel('Percentage Increase in Accidents')
plt.title('')

plt.xticks(all_estimates['start_time']) # Ensure all x-axis values are shown
plt.grid(axis='y', alpha=0.5) # Add a subtle grid for better readability
#plt.ylim([0, 1])  # Set y-axis limits if needed (e.g. 0 to 1)

plt.tight_layout() # Adjust layout to prevent labels from overlapping
ax = plt.gca()  # Get the current axis
#ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))

plt.savefig(f"{home_directory}impact_of_changing_time_dst_first_{num_weeks_diff}_weeks_feols_{hour_range}_hours.png", dpi = 300)



###############     ST separate     ############################



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



#    fit = pf.fepois("total_accidents ~ two_weeks_after_change + pre_treatment | kaza_ili + year", data = all_zoomed_in_data_agg, vcov={"CRV1": "kaza_ili"})
    fit = pf.feols("total_accidents ~ two_weeks_after_change + post_treatment | kaza_ili + year", data = all_zoomed_in_data_agg, vcov={"CRV1": "kaza_ili"})
    fit.summary()
    temp = pd.DataFrame({
        "treat_coefficient": [-float(fit.coef()["post_treatment"])],
        "treat_se": [float(fit.se()["post_treatment"])],
        "treat_pvalue": [float(fit.pvalue()["post_treatment"])],
        "two_weeks_coefficient": [float(fit.coef()["two_weeks_after_change"])],
        "two_weeks_se": [float(fit.se()["two_weeks_after_change"])],
        "two_weeks_pvalue": [float(fit.pvalue()["two_weeks_after_change"])], 
        "start_time": [start_hour], 
        "end_time": [end_hour]

    }
    ).assign(**{"significant": lambda x: x["treat_pvalue"] < 0.05})
    all_estimates = pd.concat([all_estimates, temp], ignore_index= True)



errors = 2 * all_estimates['treat_se']


plt.figure(figsize=(15, 6))  # Adjust figure size for better readability if needed
plt.bar(all_estimates['start_time'], all_estimates['treat_coefficient'] * 100, yerr=errors * 100, capsize=3)

# Add labels and title
plt.xlabel('Hour')
plt.ylabel('Percentage Increase in Accidents')
plt.title('')

plt.xticks(all_estimates['start_time']) # Ensure all x-axis values are shown
plt.grid(axis='y', alpha=0.5) # Add a subtle grid for better readability
#plt.ylim([0, 1])  # Set y-axis limits if needed (e.g. 0 to 1)

plt.tight_layout() # Adjust layout to prevent labels from overlapping
ax = plt.gca()  # Get the current axis
#ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f%%'))

plt.savefig(f"{home_directory}impact_of_changing_time_st_first_{num_weeks_diff}_weeks_feols_{hour_range}_hours.png", dpi = 300)




