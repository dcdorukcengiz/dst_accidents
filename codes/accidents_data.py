%reset -f

import numpy as np
import pandas as pd
import janitor
import matplotlib.pyplot as plt
import pyfixest as pf
from datetime import time



accidents_2019 = pd.read_excel(f"data/KADİR HAS ÜNİVERSİTESİ KAZA ARAÇ.xlsx", sheet_name = 'KAZA 2019')
accidents_2013_2019 = pd.read_excel(f"data/KADİR HAS ÜNİVERSİTESİ KAZA ARAÇ.xlsx", sheet_name = 'KAZA 2013-2018')

accidents_2019 = accidents_2019.rename(columns = { 
                                "KazaAyı": "Kaza_Ayı", 
                                "Kazaİli": "Kaza_İli"})

accidents_2013_2019 = accidents_2013_2019.rename(columns = { 
    "KazaYılı": "KazaYili"
})




combined_accidents = pd.concat([accidents_2013_2019, accidents_2019], ignore_index = True)
combined_accidents

combined_accidents_cleaner = (combined_accidents.
    assign(**{"KazaTarihi_full": lambda x: x["KazaTarihi"].astype(str) + "-" + x["Kaza_SaatDakika"]   }).
    assign(**{"KazaTarihi_full": lambda x: pd.to_datetime(x["KazaTarihi_full"])  }).
    clean_names().
    loc[:, ["kazaid", "kazayili", "kaza_ayı", "kazatarihi", "kazatarihi_full", "kaza_saatdakika", "kaza_ili", "sonuc_olumlu", "sonuc_yaralamali"]]
    
    )

combined_accidents_cleaner.to_parquet(f"data/combined_accidents_cleaner_table.parquet", index = False)
combined_accidents_cleaner = pd.read_parquet(f"data/combined_accidents_cleaner_table.parquet")


total_car_accidents_per_year = (combined_accidents_cleaner.groupby(["kazayili"]).agg(total_accidents_per_year = ('kazaid', 'count')).reset_index())

fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    total_car_accidents_per_year["kazayili"],
    total_car_accidents_per_year["total_accidents_per_year"],
    color="orange",
    marker="o",       # Adds dots
    markersize=5,     # Optional: controls dot size
    linewidth=2       # Optional: thicker line
)

ax.set_ylim(bottom=100_000)

plt.tight_layout()
fig.savefig("figures/all_accidents_trend.pdf")

num_weeks_diff = 4
combined_accidents_cleaner_w_dst = (combined_accidents_cleaner.
assign(**{"summer_hours": lambda x: ((x["kazatarihi_full"] >= '2013-03-31') & (x["kazatarihi_full"] <= '2013-10-27')) | 
        ((x["kazatarihi_full"] >= '2014-03-30') & (x["kazatarihi_full"] <= '2014-10-26')) | 
        ((x["kazatarihi_full"] >= '2015-03-29') & (x["kazatarihi_full"] <= '2015-11-08')) | 
        ((x["kazatarihi_full"] >= '2016-03-27'))}
       
).
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
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2017-03-31') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2017-03-31'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2017-10-27') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2017-10-27'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2018-03-30') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2018-03-30'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2018-10-26') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2018-10-26'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2019-03-29') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2019-03-29'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2019-11-08') & (x["kazatarihi_full"] + pd.Timedelta(weeks = num_weeks_diff) >= '2019-11-08'))      
    }).
assign(**{"placebo_two_weeks_after_change": lambda x: 
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2017-03-31') & (x["kazatarihi_full"] >= '2017-03-31'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2017-10-27') & (x["kazatarihi_full"] >= '2017-10-27'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2018-03-30') & (x["kazatarihi_full"] >= '2018-03-30'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2018-10-26') & (x["kazatarihi_full"] >= '2018-10-26'))  |
    ((x["kazatarihi_full"] - pd.Timedelta(weeks = num_weeks_diff) <= '2019-03-29') & (x["kazatarihi_full"] >= '2019-03-29'))  |
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

fit = pf.fepois("total_accidents ~ two_weeks_after_change + post_treatment | kaza_ili + year", data = all_zoomed_in_data_agg, vcov={"CRV1": "kaza_ili"})
fit.summary()


fit._vcov
all_zoomed_in_data_agg.query("pre_treatment == 1")["total_accidents"].mean()
#var(A) + var(B) - 2*cov(A,B)
#np.sqrt(8.39337799 + 7.09331279  - 2* 5.57593278)

fit.ccv()
all_zoomed_in_data_agg.info()
#################        Fatal accidents      ####################


zoomed_in_data_agg_fatal = (combined_accidents_cleaner_w_dst.query("two_weeks_before_after_change == True").query("sonuc_olumlu == 1")
.query("accident_time <= @end_time and accident_time >= @start_time")
.assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).groupby(["year", "kaza_ili", "two_weeks_after_change"]).
agg(total_accidents = ('kazaid', 'count')).reset_index().query("year < 2017"))

#pf.("total_accidents ~ two_weeks_after_change", data = zoomed_in_data_agg).summary()

placebo_zoomed_in_data_agg_fatal = (combined_accidents_cleaner_w_dst.query("placebo_two_weeks_before_after_change == True").query("sonuc_olumlu == 1").
query("accident_time <= @end_time and accident_time >= @start_time"). 
assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).groupby(["year", "kaza_ili", "placebo_two_weeks_after_change"]).
agg(total_accidents = ('kazaid', 'count')).reset_index().query("year >= 2017"))

#pf.("total_accidents ~ placebo_two_weeks_after_change", data = placebo_zoomed_in_data_agg).summary()


to_balance_data = (combined_accidents_cleaner_w_dst[["kaza_ili"]].drop_duplicates().
    merge(combined_accidents_cleaner_w_dst["kazatarihi_full"].dt.year.drop_duplicates().to_frame().rename(columns = {"kazatarihi_full": "year"}), how = "cross").
    merge(combined_accidents_cleaner_w_dst[["two_weeks_after_change"]].drop_duplicates(), how = "cross")
)


all_zoomed_in_data_agg_fatal = (pd.concat([
    zoomed_in_data_agg_fatal, placebo_zoomed_in_data_agg_fatal.rename(columns = {"placebo_two_weeks_before_after_change": "two_weeks_before_after_change", 
    "placebo_two_weeks_after_change": "two_weeks_after_change"})
]).
merge(to_balance_data, how = "right", on = ["kaza_ili", "year", "two_weeks_after_change"]).
assign(**{"post_treatment": lambda x: x["year"].case_when(
    [
        ((x["year"] >= 2017) & (x["two_weeks_after_change"] == True), 1.),
        (pd.Series(True), 0.)
    ]
) }).
assign(**{"total_accidents": lambda x: x["total_accidents"].fillna(0)}).
assign(**{"real_part": lambda x: x["year"] < 2017}).
assign(**{"kaza_ili2": lambda x: x["kaza_ili"].astype(str) + x["real_part"].astype(str)})


)


pf.feols("total_accidents ~ two_weeks_after_change + post_treatment | kaza_ili + year", data = all_zoomed_in_data_agg_fatal).summary()


#Non-fatal accident



zoomed_in_data_agg_non_fatal = (combined_accidents_cleaner_w_dst.query("two_weeks_before_after_change == True").query("sonuc_yaralamali == 1")
.query("accident_time <= @end_time and accident_time >= @start_time")
.assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).groupby(["year", "kaza_ili", "two_weeks_after_change"]).
agg(total_accidents = ('kazaid', 'count')).reset_index().query("year < 2017"))

#pf.("total_accidents ~ two_weeks_after_change", data = zoomed_in_data_agg).summary()

placebo_zoomed_in_data_agg_non_fatal = (combined_accidents_cleaner_w_dst.query("placebo_two_weeks_before_after_change == True").query("sonuc_yaralamali == 1").
query("accident_time <= @end_time and accident_time >= @start_time"). 
assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).groupby(["year", "kaza_ili", "placebo_two_weeks_after_change"]).
agg(total_accidents = ('kazaid', 'count')).reset_index().query("year >= 2017"))

#pf.("total_accidents ~ placebo_two_weeks_after_change", data = placebo_zoomed_in_data_agg).summary()


to_balance_data = (combined_accidents_cleaner_w_dst[["kaza_ili"]].drop_duplicates().
    merge(combined_accidents_cleaner_w_dst["kazatarihi_full"].dt.year.drop_duplicates().to_frame().rename(columns = {"kazatarihi_full": "year"}), how = "cross").
    merge(combined_accidents_cleaner_w_dst[["two_weeks_after_change"]].drop_duplicates(), how = "cross")
)


all_zoomed_in_data_agg_non_fatal = (pd.concat([
    zoomed_in_data_agg_non_fatal, placebo_zoomed_in_data_agg_non_fatal.rename(columns = {"placebo_two_weeks_before_after_change": "two_weeks_before_after_change", 
    "placebo_two_weeks_after_change": "two_weeks_after_change"})
]).
merge(to_balance_data, how = "right", on = ["kaza_ili", "year", "two_weeks_after_change"]).
assign(**{"post_treatment": lambda x: x["year"].case_when(
    [
        ((x["year"] >= 2017) & (x["two_weeks_after_change"] == True), 1.),
        (pd.Series(True), 0.)
    ]
) }).
assign(**{"total_accidents": lambda x: x["total_accidents"].fillna(0)}).
assign(**{"real_part": lambda x: x["year"] < 2017}).
assign(**{"kaza_ili2": lambda x: x["kaza_ili"].astype(str) + x["real_part"].astype(str)})


)


pf.feols("total_accidents ~ two_weeks_after_change + post_treatment | kaza_ili + year", data = all_zoomed_in_data_agg_non_fatal).summary()


#################            Pedestrian accidents       ##############################

pedestrian_accidents = pd.read_excel(f"data/KADİR HAS ÜNİVERSİTESİ_YOLCU YAYA.xlsx", sheet_name=None)
pedestrian_accidents.pop("VERİ BİLGİSİ", None)
pedestrian_accidents["YOYA 2019"] = pedestrian_accidents["YOYA 2019"].rename(columns = {"KazaYili": "KazaYılı"})
pedestrian_accidents_all_together = pd.concat(pedestrian_accidents, ignore_index = True).rename(columns = { 
    "KazaYılı": "KazaYili"
}).clean_names()
pedestrian_accidents_all_together.head()
only_pedestrian = pedestrian_accidents_all_together.query("yoya_kazazede == '2-Yaya'").reset_index(drop = True)
only_pedestrian_w_dates = (only_pedestrian[["kazayili", "kazaid"]].drop_duplicates().
merge(combined_accidents_cleaner_w_dst[["kazayili", "kazaid", "kazatarihi_full", 
        "two_weeks_before_after_change", "two_weeks_after_change", "placebo_two_weeks_before_after_change", 
        "placebo_two_weeks_after_change", "accident_time", "kaza_ili"]], 
on = ["kazayili", "kazaid"], how = "outer", indicator = True).query("_merge == 'both'")
)



###### Regression ready data

zoomed_in_data_agg = (only_pedestrian_w_dates.query("two_weeks_before_after_change == True")
.query("accident_time <= @end_time and accident_time >= @start_time")
.assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).groupby(["year", "kaza_ili", "two_weeks_after_change"]).
agg(total_accidents = ('kazaid', 'count')).reset_index())

#pf.("total_accidents ~ two_weeks_after_change", data = zoomed_in_data_agg).summary()

placebo_zoomed_in_data_agg = (only_pedestrian_w_dates.query("placebo_two_weeks_before_after_change == True").
query("accident_time <= @end_time and accident_time >= @start_time"). 
assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).groupby(["year", "kaza_ili", "placebo_two_weeks_after_change"]).
agg(total_accidents = ('kazaid', 'count')).reset_index())

#pf.("total_accidents ~ placebo_two_weeks_after_change", data = placebo_zoomed_in_data_agg).summary()

to_balance_data = (combined_accidents_cleaner_w_dst[["kaza_ili"]].drop_duplicates().
    merge(combined_accidents_cleaner_w_dst["kazatarihi_full"].dt.year.drop_duplicates().to_frame().rename(columns = {"kazatarihi_full": "year"}), how = "cross").
    merge(combined_accidents_cleaner_w_dst[["two_weeks_after_change"]].drop_duplicates(), how = "cross")
)


all_zoomed_in_data_agg_ped = (pd.concat([
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
assign(**{"total_accidents": lambda x: x["total_accidents"].fillna(0)}).
assign(**{"real_part": lambda x: x["year"] < 2017}).
assign(**{"cluster_var": lambda x: x["year"].astype(str) + "-" + x["two_weeks_after_change"].astype(str)})


)



fit = pf.feols("total_accidents ~ two_weeks_after_change + post_treatment | kaza_ili + year", data = all_zoomed_in_data_agg_ped)
fit.summary()


###############         non-pedestrian accidents        ##################


non_pedestrian_accidents_w_dates = (only_pedestrian[["kazayili", "kazaid"]].drop_duplicates().
merge(combined_accidents_cleaner_w_dst[["kazayili", "kazaid", "kazatarihi_full", 
        "two_weeks_before_after_change", "two_weeks_after_change", "placebo_two_weeks_before_after_change", 
        "placebo_two_weeks_after_change", "accident_time", "kaza_ili"]], 
on = ["kazayili", "kazaid"], how = "outer", indicator = True).query("_merge == 'right_only'").drop(columns = ["_merge"])
)



###### Regression ready data

zoomed_in_data_agg = (non_pedestrian_accidents_w_dates.query("two_weeks_before_after_change == True")
.query("accident_time <= @end_time and accident_time >= @start_time")
.assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).groupby(["year", "kaza_ili", "two_weeks_after_change"]).
agg(total_accidents = ('kazaid', 'count')).reset_index())

#pf.("total_accidents ~ two_weeks_after_change", data = zoomed_in_data_agg).summary()

placebo_zoomed_in_data_agg = (non_pedestrian_accidents_w_dates.query("placebo_two_weeks_before_after_change == True").
query("accident_time <= @end_time and accident_time >= @start_time"). 
assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).groupby(["year", "kaza_ili", "placebo_two_weeks_after_change"]).
agg(total_accidents = ('kazaid', 'count')).reset_index())

#pf.("total_accidents ~ placebo_two_weeks_after_change", data = placebo_zoomed_in_data_agg).summary()

to_balance_data = (combined_accidents_cleaner_w_dst[["kaza_ili"]].drop_duplicates().
    merge(combined_accidents_cleaner_w_dst["kazatarihi_full"].dt.year.drop_duplicates().to_frame().rename(columns = {"kazatarihi_full": "year"}), how = "cross").
    merge(combined_accidents_cleaner_w_dst[["two_weeks_after_change"]].drop_duplicates(), how = "cross")
)


all_zoomed_in_data_agg_non_ped = (pd.concat([
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
assign(**{"total_accidents": lambda x: x["total_accidents"].fillna(0)}).
assign(**{"real_part": lambda x: x["year"] < 2017}).
assign(**{"cluster_var": lambda x: x["year"].astype(str) + "-" + x["two_weeks_after_change"].astype(str)})


)



fit = pf.feols("total_accidents ~ two_weeks_after_change + post_treatment | kaza_ili + year", data = all_zoomed_in_data_agg_non_ped)
fit.summary()

fit2 = pf.fepois("total_accidents ~ two_weeks_after_change + post_treatment | kaza_ili + year", data = all_zoomed_in_data_agg_non_ped)
fit2.summary()




