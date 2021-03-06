#!/usr/bin/env python3
# -*- coding: utf8 -*-

# SPDX-FileCopyrightText: 2021 UdS AES <https://www.uni-saarland.de/lehrstuhl/frey.html>
# SPDX-License-Identifier: MIT


"""Demonstrate performance of SIMaaS-implementation using async IO."""

import asyncio
import copy
import csv
import json
import math
import os
import random
import sys
import uuid

import aiohttp
import numpy as np
import pandas as pd
import pendulum
import pydash
import requests
import scipy.io as sio
from codetiming import Timer
from deap import base, creator, tools
from hashids import Hashids
from invoke import task
from loguru import logger

EXIT_ENVVAR_MISSING = 1
JSON_DUMPS_INDENT = 2  # intendation width for stringified JSON in spaces

# Configure logging
log_level = os.getenv("UC1D_LOG_LEVEL", "INFO")
logger.remove()
logger.level("REQUEST", no=15, color="<light-magenta>")
logger.add(sys.stdout, level=log_level, diagnose=True, backtrace=False, enqueue=True)


# Utility functions ####################################################################
def timeseries_array_as_df(body: dict):
    """Parse timeseries-objects as pd.DataFrame."""

    tmp_dfs = []
    for result in body["data"]:
        tmp = pd.read_json(json.dumps(result["timeseries"]))
        tmp = tmp.set_index("timestamp")
        tmp[f"{result['label']}_unit"] = result["unit"]
        tmp.rename(columns={"value": result["label"]}, inplace=True)
        tmp.sort_values("timestamp", axis="index", inplace=True)

        if "datetime" in tmp.columns:
            tmp.drop(columns="datetime", inplace=True)
        logger.trace(f"tmp\n{tmp}")

        tmp_dfs.append(tmp)

    if len(tmp_dfs) > 1:
        df = tmp_dfs[0].join(tmp_dfs[1:], how="outer")
    else:
        df = tmp_dfs[0]

    logger.trace(f"df\n{df}")
    return df


def dataframe_to_json(df, time_is_epoch=True):
    """Render JSON-representation of DataFrame."""
    df.index.name = "datetime"

    logger.trace("df:\n{}".format(df))

    ts_value_objects = json.loads(
        df.to_json(orient="table")
        .replace("datetime", "timestamp")
        .replace(df.columns[0], "value")
    )["data"]
    if time_is_epoch is True:
        for x in ts_value_objects:
            x["datetime"] = pendulum.parse(x["timestamp"]).isoformat()
            x["timestamp"] = int(pendulum.parse(x["timestamp"]).format("x"))

    return ts_value_objects


def list_of_tuples_to_df(list_of_tuples):
    ids = []
    frames = []
    for id, body in list_of_tuples:
        ids.append(str(id))
        frames.append(timeseries_array_as_df(body))

    df = pd.concat(frames, keys=ids, names=["model-run", "timestamp"])
    df.sort_values(by=["model-run", "timestamp"], axis="index", inplace=True)
    logger.trace(f"df\n{df}")

    return df


def add_model(filepath, origin, params, model_name):
    url = f"{origin}/models"
    files = {"fmu": ("model.fmu", open(filepath, "rb"), "application/octet-stream")}

    logger.info(f"Adding model <{model_name}.fmu> to SIMaaS-instance...")
    logger.log("REQUEST", f"POST {url}")

    r = requests.post(url, params=params, files=files)

    location = r.headers["Location"]

    return location


# Get (fake) ensemble weather forecast #################################################
async def request_weather_forecast(
    session: aiohttp.ClientSession, station_id: str, params: dict, wfc_raw: list
):
    try:
        origin = os.environ["UC1D_WEATHER_API_ORIGIN"]
    except KeyError as e:
        logger.critical(f"Required ENVVAR {e} is not set; exciting.")
        sys.exit(EXIT_ENVVAR_MISSING)

    href = f"{origin}/weather-stations/{station_id}/forecast"
    logger.debug(
        f"Requesting model run {params['model-run']} of {params['model'].upper()}-model for station {station_id}..."
    )

    async with session.get(href, params=params) as res:
        logger.log("REQUEST", f"GET {res.url}")
        rep = await res.json()
        logger.trace(json.dumps(rep, indent=JSON_DUMPS_INDENT))

        wfc_raw.append((params["model-run"], rep))


async def weather_forecasts_as_df(start: int, end: int):
    station_id = 10708
    model = "cosmo-d2"
    model_runs = (
        ["03", "09", "15", "21"]
        if model.upper() == "MOSMIX"
        else [
            "00",
            "03",
            "06",
            "09",
            "12",
            "15",
            "18",
            "21",
        ]
    )
    quantities = [
        "t_2m",
        "aswdir_s",
        "aswdifd_s",
        "ws_10m",
    ]
    api_token = os.getenv("UC1D_WEATHER_API_JWT")

    # Get ensemble forecast for weather at a DWD weather station; put data into queue
    q_nwp = []
    headers = {"Authorization": f"Bearer {api_token}"} if api_token else None
    async with aiohttp.ClientSession(headers=headers) as session:
        weather_forecasts = [
            asyncio.create_task(
                request_weather_forecast(
                    session,
                    station_id,
                    {
                        "model": model,
                        "model-run": model_run,
                        "quantities": ",".join(quantities),
                        "from": start,
                        "to": end,
                    },
                    q_nwp,
                )
            )
            for model_run in model_runs
        ]

        logger.info("Requesting weather forecasts...")
        await asyncio.gather(*weather_forecasts)

    # Append all elements in list to dataframe
    df = list_of_tuples_to_df(q_nwp)

    return df


# Prepare simulation requests for running the ensemble PV forecast #####################
async def get_simulation_request_bodies():
    period = pendulum.period(
        pendulum.datetime(2020, 4, 17).start_of("day"),
        pendulum.datetime(2020, 4, 17).end_of("day"),
    )  # TODO make configurable
    start = int(period.start.format("x"))
    end = int(period.end.format("x"))
    output_interval = 900

    logger.info(
        f"Loading all ensemble forecasts for the time from {period.start.isoformat()} to {period.end.isoformat()}..."
    )

    # Fetch weather forecasts and store in MultiIndex-dataframe
    outfile = "./weather_forecast.csv"  # FIXME
    if os.getenv("UC1D_WEATHER_FROM_DISK", "true") == "true":
        # Alternatively, load weather data from disk to save time during development
        df = pd.read_csv(
            outfile,
            index_col=[0, 1],
            parse_dates=["timestamp"],
        )
        df.sort_values(by=["model-run", "timestamp"], axis="index", inplace=True)
    else:
        df = await weather_forecasts_as_df(start, end)
        df.to_csv(outfile, quoting=csv.QUOTE_NONNUMERIC)

    with pd.option_context("display.max_rows", None):
        logger.trace(f"df\n{df}")

    # Ensure that the origin of the API is provided
    for envvar in ["UC1D_SIMAAS_ORIGIN", "UC1D_EFC_FMU"]:
        try:
            os.environ[envvar]
        except KeyError as e:
            logger.critical(f"Required ENVVAR {e} is not set; exciting.")
            sys.exit(EXIT_ENVVAR_MISSING)

    simaas_origin = os.environ["UC1D_SIMAAS_ORIGIN"]

    # Add model to API
    fmu_path = os.environ["UC1D_EFC_FMU"]
    record_component_names = [
        "irradianceTemperatureWindSpeed2Power.plantRecord",
        "irradianceTemperatureWindSpeed2Power.location",
    ]
    payload = {"records": ",".join(record_component_names)}
    model_name = "PhotoVoltaicPowerPlantFMU"
    model_url = add_model(fmu_path, simaas_origin, payload, model_name)
    model_id = model_url.split("/")[-1]

    # Create model instance to be used
    instances_to_create = [
        (
            f"{model_url}/instances",
            {
                "modelName": model_name,
                "parameters": {
                    "latitude": {
                        "value": 49.31659,
                        "unit": "deg",
                    },
                    "longitude": {
                        "value": 6.749953,
                        "unit": "deg",
                    },
                    "elevation": {
                        "value": 181,
                        "unit": "m",
                    },
                    "panelArea": {
                        "value": 0.156 * 0.156 * 60 * 156,
                        "unit": "m2",
                    },
                    "plantEfficiency": {
                        "value": 0.17,
                        "unit": "1",
                    },
                    "T_cell_ref": {
                        "value": 25,
                        "unit": "degC",
                    },
                    "panelTilt": {
                        "value": 28,
                        "unit": "deg",
                    },
                    "panelAzimuth": {
                        "value": 47,
                        "unit": "deg",
                    },
                    "environmentAlbedo": {
                        "value": 0.2,
                        "unit": "1",
                    },
                    "nsModule": {
                        "value": 6,
                        "unit": "1",
                    },
                    "npModule": {
                        "value": 26,
                        "unit": "1",
                    },
                },
            },
        )
    ]

    logger.trace(json.dumps(instances_to_create, indent=JSON_DUMPS_INDENT))

    q_loc = []  # container for hrefs to newly created model instances

    # Create model instances for all individuals; retrieve their URLs
    instantiating = [
        asyncio.create_task(create_instance(h, b, q_loc))
        for h, b in instances_to_create
    ]
    await asyncio.gather(*instantiating)

    # Construct request bodies from dataframe
    bc = {}
    for model_run in df.index.levels[0]:
        tmp = df.loc[model_run]
        tmp = tmp.interpolate()

        start = tmp.first_valid_index().timestamp() * 1000
        model_instance_url = q_loc[0]

        # Create request-body
        request_body = {
            "modelId": model_id,
            "simulationParameters": {
                "startTime": start,
                "stopTime": end,
                "inputTimeIsRelative": False,
                "outputInterval": output_interval,
            },
            "inputTimeseries": [],
        }

        # Update request_body with timeseries of temp_air, irr_diffuse and irr_direct
        map = {
            "t_2m": "temperature",
            "aswdir_s": "directHorizontalIrradiance",
            "aswdifd_s": "diffuseHorizontalIrradiance",
            "ws_10m": "windSpeed",
        }
        for q in map.keys():
            request_body["inputTimeseries"].append(
                {
                    "label": map[q],
                    "unit": tmp[f"{q}_unit"][0],
                    "timeseries": dataframe_to_json(tmp[[q]]),
                }
            )

        bc[model_run] = {
            "href": f"{model_instance_url}/experiments",
            "body": request_body,
        }

    logger.trace(json.dumps(bc, indent=JSON_DUMPS_INDENT))

    return bc


# Asynchronously request simulations to run and await results ##########################
async def create_instance(href: str, body: dict, q_loc: list):

    headers = None
    model_id = href.split("/")[-2]

    logger.info(f"Creating new instance of model <{model_id}>...")
    logger.log("REQUEST", f"POST {href}")

    async with aiohttp.ClientSession() as session:
        async with session.post(href, json=body, headers=headers) as res:
            href_location = res.headers["Location"]
            logger.trace(f"Location: {href_location}")

            # Enqueue link to resource just created
            q_loc.append(href_location)


async def request_simulation(
    session: aiohttp.ClientSession, id: str, href: str, body: dict, q: asyncio.Queue
):
    """Request simulation by POSTing `body` to given `href`."""

    iid = body["modelId"]
    req_id = str(uuid.uuid4())
    headers = {"X-Request-Id": req_id}

    logger.info(f"Requesting simulation of model <{iid}>...")
    logger.log("REQUEST", f"POST {href}")

    # Trigger simulation and wait for 201
    async with session.post(href, json=body, headers=headers) as res:
        href_location = res.headers["Location"]
        logger.trace(f"Location: {href_location}")

        # Enqueue link to resource just created
        await q.put((id, req_id, href_location))


async def poll_until_done(
    session: aiohttp.ClientSession, q1: asyncio.Queue, q2: asyncio.Queue
):
    """Poll specific experiment until it's done or failed."""

    counter_max = int(os.getenv("UC1D_POLLING_RETRIES", "30"))
    freq = float(os.getenv("UC1D_POLLING_FREQUENCY", "0.1"))

    while True:
        # Retrieve first item from queue
        id, req_id, href = await q1.get()
        headers = {"X-Request-Id": req_id}

        # Poll status of simulation
        counter = 0
        href_result = None
        while counter < counter_max:
            if counter > (counter_max / 2):
                freq *= 2
            logger.log("REQUEST", f"GET {href}")
            async with session.get(href, headers=headers) as res:
                rep = await res.json()
                status = rep["status"]
                logger.trace(
                    f"Polling status of simulation for individual '{id}': {status}"
                )

                if status == "DONE":
                    href_result = rep["linkToResult"]
                    break
                if status == "FAILED":
                    logger.warning("Simulation failed")
                    break

                counter += 1
                await asyncio.sleep(freq)

        # Enqueue link to result
        await q2.put((id, req_id, href_result))

        # Indicate that a formerly enqueued task is complete
        q1.task_done()


async def fetch_simulation_result(
    session: aiohttp.ClientSession, q: asyncio.Queue, q_repr_all: list
):
    """Get the simulation result and parse it as dataframe."""

    while True:
        # Retrieve first item from queue
        id, req_id, href = await q.get()
        headers = {"X-Request-Id": req_id}

        # Get simulation result
        logger.info(f"Retrieving result of simulation for individual '{id}''")
        async with session.get(href, headers=headers) as res:
            logger.log("REQUEST", f"GET {href}")
            rep = await res.json()
            logger.trace(json.dumps(rep, indent=JSON_DUMPS_INDENT))

            # Enqueue for post-processing
            q_repr_all.append((id, rep))

        # Indicate that a formerly enqueued task is complete
        q.task_done()


async def await_collection_of_simulatons(dict_id_href_body: dict):
    count_simulations_total = len(dict_id_href_body)

    # Get all simulation results and enqueue the representations for post-processing
    q_repr_all = []
    session = aiohttp.ClientSession()

    # Spawn task queues, producer and workers
    q_sim = asyncio.Queue()
    q_res = asyncio.Queue()

    requesting = [
        asyncio.create_task(
            request_simulation(session, id, r["href"], r["body"], q_sim)
        )
        for id, r in dict_id_href_body.items()
    ]
    polling = [
        asyncio.create_task(poll_until_done(session, q_sim, q_res))
        for n in range(count_simulations_total)
    ]
    fetching = [
        asyncio.create_task(fetch_simulation_result(session, q_res, q_repr_all))
        for n in range(count_simulations_total)
    ]

    # Await addition of all tasks to the first queue
    logger.info("Awaiting the completion of all simulations...")
    await asyncio.gather(*requesting)

    # Wait until the queues are fully processed; then cancel worker tasks, close session
    await q_sim.join()
    await q_res.join()

    for task in polling + fetching:
        task.cancel()

    await session.close()

    return q_repr_all


# Demo 1: many simulations of the same model instance ##################################
@logger.catch
async def ensemble_forecast():
    """Execute ensemble forecast and represent as dataframe."""

    # Assemble data used as input for simulations ######################################
    timer_wfc = Timer(
        text="Overall duration for preparing weather forecasts: {:.2f} seconds",
        logger=logger.success,
    )
    with timer_wfc:
        dict_id_href_body = await get_simulation_request_bodies()

    # Await the completion of all simulations in collection
    timer_overall = Timer(
        text="Overall duration for running all simulations: {:.2f} seconds",
        logger=logger.success,
    )

    with timer_overall:
        q_repr_all = await await_collection_of_simulatons(dict_id_href_body)

    return q_repr_all


@task
def demo_efc(ctx):
    # Run ensemble forecast
    q_repr_all = asyncio.run(ensemble_forecast())

    # Assemble all simulation results in a MultiIndex-dataframe
    df = list_of_tuples_to_df(q_repr_all)

    # Create and show very simple plot
    logger.info("Creating and showing simple run plot of ensemble forecast...")
    ids = []
    frames = []
    key = "powerDC"
    for run in df.index.levels[0]:
        ids.append(run)
        frames.append(df.xs(run)[key])

    df1 = pd.concat(frames, axis=1, keys=ids)
    df1 = df1.sort_index(axis="columns")

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        logger.trace(f"df1\n{df1}")

    import matplotlib.pyplot as plt

    plt.close("all")
    df1.plot(figsize=(16, 9))
    plt.show()

    logger.success("Done!")


# Demo 2: simulation of many new model instances #######################################
def get_values_e24_series(lower, upper):
    """
    Return the values of the E24 series between `lower` and `upper`.

    Both `lower` and `upper` are included in the result.
    See https://en.wikipedia.org/wiki/E_series_of_preferred_numbers for details.
    """
    lower = float(lower)
    upper = float(upper)

    e24_values = [
        1.0,
        1.1,
        1.2,
        1.3,
        1.5,
        1.6,
        1.8,
        2.0,
        2.2,
        2.4,
        2.7,
        3.0,
        3.3,
        3.6,
        3.9,
        4.3,
        4.7,
        5.1,
        5.6,
        6.2,
        6.8,
        7.5,
        8.2,
        9.1,
    ]

    s = pd.Series(data=e24_values)

    powered = []
    for power in range(int(math.log(lower, 10)), int(math.log(upper, 10)) + 1):
        powered.append(s * (10 ** power))

    s = s.append(powered)
    s.sort_values(inplace=True)
    s.drop_duplicates(inplace=True)

    r = s[s.between(lower, upper, inclusive="both")]

    return r.to_numpy()


def get_component_values(component_values, ind):
    return {
        "R1": {
            "value": int(component_values["r_res"][ind[0]]),
            "unit": "Ohm",
        },
        "R2": {
            "value": int(component_values["r_res"][ind[1]]),
            "unit": "Ohm",
        },
        "R3": {
            "value": int(component_values["r_res"][ind[2]]),
            "unit": "Ohm",
        },
        "R4": {
            "value": int(component_values["r_res"][ind[3]]),
            "unit": "Ohm",
        },
        "TH1": {
            "value": int(component_values["th_res"][ind[4]]),
            "unit": "Ohm",
        },
        "TH2": {
            "value": int(component_values["th_res"][ind[5]]),
            "unit": "Ohm",
        },
        "B1": {
            "value": int(component_values["th_beta"][ind[6]]),
            "unit": "1/K",
        },
        "B2": {
            "value": int(component_values["th_beta"][ind[7]]),
            "unit": "1/K",
        },
    }


async def evaluate_generation(evaluate, generation, component_values, model_url):
    """Asynchronously evaluate fitness for each individual in generation."""

    for ind in generation:
        logger.trace(f"{ind} => __undefined__")

    # Build list of request parts for creating instances
    model_id = model_url.split("/")[-1]
    model_name = "TemperatureCompensationFMU"
    instances_to_create = []
    for ind in generation:
        instances_to_create.append(
            (
                f"{model_url}/instances",
                {
                    "modelName": model_name,
                    "parameters": get_component_values(component_values, ind),
                },
            )
        )

    logger.trace(json.dumps(instances_to_create, indent=JSON_DUMPS_INDENT))

    q_loc = []  # container for hrefs to newly created model instances

    # Create model instances for all individuals; retrieve their URLs
    instantiating = [
        asyncio.create_task(create_instance(h, b, q_loc))
        for h, b in instances_to_create
    ]
    await asyncio.gather(*instantiating)

    logger.trace(json.dumps(q_loc, indent=JSON_DUMPS_INDENT))

    # Assemble hrefs and request bodies for simulation of individuals
    tmp = pd.DataFrame(
        data={"temperature": list(range(0, 61, 1))}, index=list(range(0, 61, 1))
    )
    tmp["temperature"] = 2 * tmp.index - 40
    request_body = {
        "modelId": model_id,
        "simulationParameters": {
            "startTime": 0,
            "stopTime": 60,
            "inputTimeIsRelative": True,
            "outputInterval": 1,
        },
        "inputTimeseries": [
            {
                "label": "temperature",
                "unit": "K",
                "timeseries": dataframe_to_json(tmp, time_is_epoch=False),
            }
        ],
    }

    logger.trace(json.dumps(request_body, indent=JSON_DUMPS_INDENT))

    dict_id_href_body = {}
    hashids = Hashids()
    for ind, href in zip(generation, q_loc):
        hashid = hashids.encode(*ind)
        body = copy.deepcopy(request_body)
        body["modelInstanceId"] = href.split("/")[-1]
        dict_id_href_body[hashid] = {
            "href": f"{href}/experiments",
            "body": body,
        }

    logger.trace(json.dumps(dict_id_href_body, indent=JSON_DUMPS_INDENT))

    # Await the completion of all simulations in collection
    q_repr_all = await await_collection_of_simulatons(dict_id_href_body)

    # Represent all simulation results as dataframe
    df = list_of_tuples_to_df(q_repr_all)
    logger.trace(f"df\n{df}")

    ids = []
    frames = []
    key = "voltage"
    for run in df.index.levels[0]:
        ids.append(run)
        frames.append(df.xs(run).set_index(df.xs(run)["tempK"])[key])

    df = pd.concat(frames, axis=1, keys=ids)
    df = df.sort_index(axis="columns")

    df["tempC"] = df.index - 273.15
    df["reference"] = 1.125e-5 * df["tempC"] ** 2 - 1.125e-4 * df["tempC"] + 1.026e-1

    logger.trace(f"df\n{df}")

    # Amend each individual by its fitness value
    for ind in generation:
        hashid = hashids.encode(*ind)
        fitness = ((df[hashid] - df["reference"]) ** 2).mean() ** 0.5
        ind.fitness.values = (fitness,)  # MUST assign tuple!
        logger.debug(f"{hashid}: {ind} => {ind.fitness.values[0]}")


@logger.catch
async def genetic_algorithm():
    """
    Use genetic algorithm to solve component selection problem.

    https://www.edn.com/genetic-algorithm-solves-thermistor-network-component-values
    https://de.mathworks.com/videos/optimal-component-selection-using-the-mixed-integer-
    genetic-algorithm-68956.html

    https://deap.readthedocs.io/en/master/overview.html
    """

    component_values = {
        "r_res": get_values_e24_series(300, 220000),
        "th_res": np.array(
            [
                50,
                220,
                1000,
                2200,
                3300,
                4700,
                10000,
                22000,
                33000,
            ]
        ),
        "th_beta": np.array(
            [
                2750,
                3680,
                3560,
                3620,
                3528,
                3930,
                3960,
                4090,
                3740,
            ]
        ),
    }

    # Add model to API
    for envvar in ["UC1D_SIMAAS_ORIGIN", "UC1D_GA_FMU"]:
        try:
            os.environ[envvar]
        except KeyError as e:
            logger.critical(f"Required ENVVAR {e} is not set; exciting.")
            sys.exit(EXIT_ENVVAR_MISSING)
    simaas_origin = os.environ["UC1D_SIMAAS_ORIGIN"]

    fmu_path = os.environ["UC1D_GA_FMU"]
    record_component_names = ["thermistorBridge.data"]
    payload = {"records": ",".join(record_component_names)}
    model_name = "TemperatureCompensationFMU"
    model_url = add_model(fmu_path, simaas_origin, payload, model_name)

    # Set up genetic algorithm and its parameters ######################################
    POP_SIZE = 40  # number of individuals in a generation
    TOURN_SIZE = 3  # number of individuals entered into tournament
    CXPB = 0.6  # probability with which two individuals are crossed
    MUTPB = 1 / 8  # probability for mutating an individual
    NGEN = 8  # number of generations to try

    # Generate individuals that match problem representation
    def gen_idx():
        (n, n_min, n_max, m, m_min, m_max, o, o_min, o_max) = (
            4,
            0,
            len(component_values["r_res"]) - 1,
            2,
            0,
            len(component_values["th_res"]) - 1,
            2,
            0,
            len(component_values["th_beta"]) - 1,
        )
        a = [random.randint(n_min, n_max) for i in range(n)]
        b = [random.randint(m_min, m_max) for i in range(m)]
        c = [random.randint(o_min, o_max) for i in range(o)]

        return a + b + c

    # Specify type of problem and each individual
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    # Initialize first generation at random
    toolbox = base.Toolbox()
    toolbox.register(
        "individual",
        tools.initIterate,
        creator.Individual,
        gen_idx,
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the genetic operators to use
    toolbox.register("evaluate", lambda x: x)  # dummy function; registration necessary
    toolbox.register("select", tools.selTournament, tournsize=TOURN_SIZE)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=MUTPB)

    # Enable collection of statistics
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(key=len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("max", np.max)
    mstats.register("avg", np.mean)
    mstats.register("min", np.min)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "fitness"
    logbook.chapters["fitness"].header = "max", "avg", "min"

    hall_of_fame = tools.HallOfFame(3)

    # Instantiate and evaluate the first generation
    pop = toolbox.population(n=POP_SIZE)

    await evaluate_generation(
        toolbox.evaluate,
        pop,
        component_values,
        model_url,
    )
    record = mstats.compile(pop)
    logbook.record(gen=0, evals=len(pop), **record)
    hall_of_fame.update(pop)

    for g in [x + 1 for x in range(NGEN - 1)]:
        logger.success(f"^ {g} ---:::--- {g+1} v")

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # # FIXME not all missing fitness values are updated if you try this!
        # invalid_ind = pydash.uniq(invalid_ind)
        if not pydash.is_empty(invalid_ind):
            await evaluate_generation(
                toolbox.evaluate,
                invalid_ind,
                component_values,
                model_url,
            )

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Collect statistics
        record = mstats.compile(pop)
        logbook.record(gen=g, evals=len(invalid_ind), **record)
        hall_of_fame.update(pop)

    logger.success(f"\n{logbook}")
    conclusion = (
        f"Hall of Fame:\n"
        f"1. {hall_of_fame[0]} => {hall_of_fame[0].fitness.values[0]}\n"
        f"2. {hall_of_fame[1]} => {hall_of_fame[1].fitness.values[0]}\n"
        f"3. {hall_of_fame[2]} => {hall_of_fame[2].fitness.values[0]}\n"
    )
    logger.success(conclusion)

    logger.success(
        "Best parameter set found:\n{}".format(
            json.dumps(
                get_component_values(component_values, hall_of_fame[0]),
                indent=JSON_DUMPS_INDENT,
            )
        )
    )

    return logbook


@task
def demo_ga(ctx):
    logbook = asyncio.run(genetic_algorithm())

    # https://deap.readthedocs.io/en/master/tutorials/basic/part3.html
    # #some-plotting-sugar
    gen = logbook.select("gen")
    fit_mins = logbook.chapters["fitness"].select("min")
    fit_maxs = logbook.chapters["fitness"].select("max")

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, fit_maxs, "r-", label="Maximum Fitness")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="center right")

    plt.show()
