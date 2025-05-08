rm(list=ls())
library(fect)
library(gsynth)
library(arrow)
library(tidyverse)

home_directory = "/Users/EzgilovesDoruk/Desktop/education_health/"
all_relative_to_dst <- read_parquet(paste0(home_directory, "all_relative_to_dst.parquet"))
all_relative_to_st <- read_parquet(paste0(home_directory, "all_relative_to_st.parquet"))

all_relative_to_dst_agg <-all_relative_to_dst |> 
  mutate(weeks = floor(relative_to_real_dst/7)) |> 
  group_by(kaza_ili, placebo_years, weeks) |> 
  count() |> 
  ungroup()

all_relative_to_st_agg <-all_relative_to_st |> 
  mutate(weeks = floor(relative_to_real_dst/7)) |> 
  group_by(kaza_ili, placebo_years, weeks) |> 
  count() |> 
  ungroup()


all_relative_agg <- bind_rows(all_relative_to_dst_agg, all_relative_to_st_agg) |> 
  group_by(kaza_ili, placebo_years, weeks) |> 
  summarise(total_accidents = sum(n), .groups = "drop") |> 
  ungroup() |> 
  mutate(kaza_ili2 = paste0(kaza_ili, "_", placebo_years)) |> 
  mutate(treatment = case_when(weeks >= 0 ~ 1,
  TRUE ~ 0)) |> 
  mutate(post_treatment = case_when(weeks >= 0 & placebo_years == FALSE ~ 1,
  TRUE ~ 0))



out <- gsynth(total_accidents ~ post_treatment, data = all_relative_agg, parallel = FALSE, 
  index = c("kaza_ili2","weeks"), force = "two-way", EM = TRUE, 
  CV = TRUE, r = c(0, 5), se = TRUE, inference = "parametric") 

out <- gsynth(total_accidents ~ post_treatment, data = all_relative_agg, parallel = FALSE, 
  index = c("kaza_ili2","weeks"), estimator = "mc",
  se = FALSE, nboots = 100, r = c(0, 5), 
  CV = TRUE, force = "two-way",  
  inference = "nonparametric")

plot(out)
