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


combined_accidents_cleaner = pd.read_parquet(f"data/combined_accidents_cleaner_table.parquet")

#Hourly Accidents

hourly_accidents = (combined_accidents_cleaner.
    assign(**{"year": lambda x: x["kazatarihi_full"].dt.year}).
    assign(**{"hour": lambda x: x["kazatarihi_full"].dt.hour}).
    groupby(["hour", "year"]).
    agg(total_accidents = ('kazaid', 'count')).
    reset_index()
    )


import matplotlib.pyplot as plt
import numpy as np

# Unique years and setup
years = hourly_accidents['year'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(years)))  # Different color palette
markers = ['o', 's', 'v', '^', '<', '>', 'D', 'p', '*', 'h']  # Add more if needed

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each year
for i, year in enumerate(years):
    year_data = hourly_accidents[hourly_accidents['year'] == year]
    ax.plot(
        year_data['hour'],
        year_data['total_accidents']/365,
        label=str(year),
        color=colors[i % len(colors)],
        marker=markers[i % len(markers)],
        linestyle='-',
        linewidth=1.8
    )

# Customize plot
ax.set_xlabel('Hour', fontsize=12)
ax.set_ylabel('# Accidents', fontsize=12)
ax.set_title('Average Daily Traffic Accidents by Hour', fontsize=14)
ax.set_xticks(range(0, 24))
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_ylim(bottom=0)
ax.legend(fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=7)

fig.tight_layout()
ax.get_yaxis().set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

fig.savefig("figures/hourly_total_accidents_per_year.pdf")
