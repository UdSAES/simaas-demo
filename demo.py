#!/usr/bin/env python3
# -*- coding: utf8 -*-

# SPDX-FileCopyrightText: 2021 UdS AES <https://www.uni-saarland.de/lehrstuhl/frey.html>
# SPDX-License-Identifier: MIT


"""Demonstrate performance of SIMaaS-implementation using async IO."""

import asyncio
import copy
import csv
import json
import os
import random
import sys
import uuid

import aiohttp
import numpy as np
import pandas as pd
import pendulum
import pydash
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
        logger.trace(f"GET {res.url}")
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
    outfile = f"/home/moritz/work/projekte/designetz/software/simaas/demo/weather_forecast.csv"
    if os.getenv("UC1D_WEATHER_FROM_DISK", "false") == "true":
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

    # Construct request bodies from dataframe
    bc = {}

    try:
        origin = os.environ["UC1D_SIMAAS_ORIGIN"]
    except KeyError as e:
        logger.critical(f"Required ENVVAR {e} is not set; exciting.")
        sys.exit(EXIT_ENVVAR_MISSING)

    href = f"{origin}/experiments"

    for model_run in df.index.levels[0]:
        tmp = df.loc[model_run]
        tmp = tmp.interpolate()

        start = tmp.first_valid_index().timestamp() * 1000

        # Create request-body
        request_body = {
            "modelInstanceID": "29f11d50-f11e-46e7-ba2f-7d69f796a101",
            "simulationParameters": {
                "startTime": start,
                "stopTime": end,
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
            "href": href,
            "body": request_body,
        }

    logger.trace(json.dumps(bc, indent=JSON_DUMPS_INDENT))

    return bc


# Asynchronously request simulations to run and await results ##########################
async def create_instance(href: str, body: dict, q_loc: list):

    headers = None
    model_id = None

    logger.info(f"Creating new instance of model {model_id}...")
    logger.trace(f"POST {href}")

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

    iid = body["modelInstanceID"]
    req_id = str(uuid.uuid4())
    headers = {"X-Request-Id": req_id}

    logger.info(f"Requesting simulation of model instance <{iid}>...")
    logger.trace(f"POST {href}")

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
            logger.trace(f"GET {href}")
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
            logger.trace(f"GET {href}")
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
        logger=logger.info,
    )
    with timer_wfc:
        dict_id_href_body = await get_simulation_request_bodies()

    # Await the completion of all simulations in collection
    q_repr_all = await await_collection_of_simulatons(dict_id_href_body)

    return q_repr_all


@task
def demo_efc(ctx):
    timer_overall = Timer(
        text="Overall duration for running ensemble forecast: {:.2f} seconds",
        logger=logger.info,
    )

    with timer_overall:
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

    logger.info("Done!")


# Demo 2: simulation of many new model instances #######################################
def get_component_values(component_values, ind):
    return {
        "R1": int(component_values["r_res"][ind[0]][0]),
        "R2": int(component_values["r_res"][ind[1]][0]),
        "R3": int(component_values["r_res"][ind[2]][0]),
        "R4": int(component_values["r_res"][ind[3]][0]),
        "TH1": int(component_values["th_res"][ind[4]][0]),
        "TH2": int(component_values["th_res"][ind[5]][0]),
        "B1": int(component_values["th_beta"][ind[6]][0]),
        "B2": int(component_values["th_beta"][ind[7]][0]),
    }


async def evaluate_generation(evaluate, generation, component_values):
    """Asynchronously evaluate fitness for each individual in generation."""

    for ind in generation:
        logger.trace(f"{ind} => __undefined__")

    # Build list of request parts for creating instances
    instances_to_create = []
    for ind in generation:
        instances_to_create.append(
            (
                "http://localhost:4000/model-instances",
                {
                    "model": "http://localhost:3100/models/65d7ce93-2804-4faa-a5dd-2fe0f24d8bae/model.fmu",
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
        "modelInstanceID": None,
        "simulationParameters": {
            "startTime": 0,
            "stopTime": 60,
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
        body["modelInstanceID"] = href.split("/")[-1]
        dict_id_href_body[hashid] = {
            "href": "http://localhost:4000/experiments",
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

    # Load possible component parameters from disk
    possible_values_file = "./Thermistor_MixedIntegerGA/StandardComponentValues.mat"
    possible_values = sio.loadmat(possible_values_file)

    component_values = {
        "r_res": possible_values["Res"],
        "th_res": possible_values["ThVal"],
        "th_beta": possible_values["ThBeta"],
    }

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
    toolbox.register("evaluate", evaluate_individual)
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

    await evaluate_generation(toolbox.evaluate, pop, component_values)
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
            await evaluate_generation(toolbox.evaluate, invalid_ind, component_values)

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

@task
def demo_ga(ctx):
    asyncio.run(genetic_algorithm())
