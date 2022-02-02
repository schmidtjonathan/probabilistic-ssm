import functools
import logging
from datetime import timedelta

import numpy as np
import pandas as pd

JHU_CSV_URL = "https://raw.githubusercontent.com/datasets/covid-19/main/data/countries-aggregated.csv"
UN_POPULATION_CSV_URL = "https://raw.githubusercontent.com/owid/covid-19-data/152b2236a32f889df3116c7121d9bb14ce2ff2a8/scripts/input/un/population_2020.csv"


def load_COVID_data(country, num_data_points=None):

    # Population
    population_df = pd.read_csv(
        UN_POPULATION_CSV_URL,
        keep_default_na=False,
        usecols=["entity", "year", "population"],
    )
    population_df = population_df.loc[population_df["entity"] == country]
    population_df = population_df.loc[population_df["year"] == 2020]
    population = float(population_df["population"])

    # COVID data
    cases_df = pd.read_csv(JHU_CSV_URL)
    cases_df["Date"] = pd.to_datetime(cases_df["Date"])
    cases_df = cases_df.loc[cases_df["Country"] == country]

    day_zero = cases_df["Date"].iloc[0]

    def count_days(date, start):
        return (date - start).days

    count_days_from_zero = functools.partial(count_days, start=day_zero)
    cases_df = cases_df.assign(CountDays=cases_df["Date"].apply(count_days_from_zero))

    # Extract cumulative (original) data
    days_from_start = cases_df["CountDays"].to_numpy()

    confirmed_cumulative = cases_df["Confirmed"].to_numpy()
    recovered_cumulative = cases_df["Recovered"].to_numpy()
    deaths_cumulative = cases_df["Deaths"].to_numpy()

    # Make data arrays immutable
    days_from_start.setflags(write=False)
    confirmed_cumulative.setflags(write=False)
    recovered_cumulative.setflags(write=False)
    deaths_cumulative.setflags(write=False)

    # Compute the daily cases from cumulative data
    confirmed_daily = np.diff(confirmed_cumulative, prepend=0)
    recovered_daily = np.diff(recovered_cumulative, prepend=0)
    deaths_daily = np.diff(deaths_cumulative, prepend=0)

    # Make data arrays immutable
    confirmed_daily.setflags(write=False)
    recovered_daily.setflags(write=False)
    deaths_daily.setflags(write=False)

    # Sanity checks
    assert np.all(np.cumsum(confirmed_daily) == confirmed_cumulative)
    assert np.all(np.cumsum(recovered_daily) == recovered_cumulative)
    assert np.all(np.cumsum(deaths_daily) == deaths_cumulative)

    date_range_x = np.array(
        [day_zero + timedelta(int(xday)) for xday in days_from_start]
    )

    logging.info(
        f"Data ranges from {date_range_x[0]:%d, %b %Y} to {date_range_x[-1 if num_data_points is None else num_data_points]:%d, %b %Y}."
    )

    D_data = deaths_cumulative
    R_data = recovered_cumulative
    I_data = confirmed_cumulative - (R_data + D_data)
    S_data = population - (I_data + R_data + D_data)

    SIRD_data = np.stack(
        [np.array(t, dtype=np.float64) for t in [S_data, I_data, R_data, D_data]]
    ).T
    SIRD_data = SIRD_data[:num_data_points]
    SIRD_data.setflags(write=False)
    return (day_zero, date_range_x, SIRD_data, population)
