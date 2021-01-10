#!/usr/bin/env python3
# -*- coding: utf8 -*-

# SPDX-FileCopyrightText: 2021 UdS AES <https://www.uni-saarland.de/lehrstuhl/frey.html>
# SPDX-License-Identifier: MIT


"""Demonstrate performance of SIMaaS-implementation using async IO."""

import asyncio
import csv
import json
import os
import sys

import aiohttp
import pandas as pd
import pendulum
from codetiming import Timer
from loguru import logger

EXIT_ENVVAR_MISSING = 1
JSON_DUMPS_INDENT = 2  # intendation width for stringified JSON in spaces

# Configure logging
log_level = os.getenv("UC1D_LOG_LEVEL", "INFO")
logger.remove()
logger.add(sys.stdout, level=log_level, diagnose=True, backtrace=False, enqueue=True)


def timeseries_array_as_df(body: dict):
    """Parse timeseries-objects as pd.DataFrame."""

    tmp_dfs = []
    for result in body["data"]:
        tmp = pd.read_json(json.dumps(result["timeseries"]))
        tmp = tmp.set_index("timestamp")
        tmp[f"{result['label']}_unit"] = result["unit"]
        tmp.rename(columns={"value": result["label"]}, inplace=True)
        logger.trace(f"tmp\n{tmp}")

        tmp_dfs.append(tmp)

    if len(tmp_dfs) > 1:
        df = tmp_dfs[0].join(tmp_dfs[1:], how="outer")
    else:
        df = tmp_dfs[0]

    logger.trace(f"df\n{df}")
    return df


def dataframe_to_json(df):
    """Render JSON-representation of DataFrame."""
    df.index.name = "datetime"

    logger.trace("df:\n{}".format(df))

    ts_value_objects = json.loads(
        df.to_json(orient="table")
        .replace("datetime", "timestamp")
        .replace(df.columns[0], "value")
    )["data"]
    for x in ts_value_objects:
        x["datetime"] = pendulum.parse(x["timestamp"]).isoformat()
        x["timestamp"] = int(pendulum.parse(x["timestamp"]).format("x"))
    return ts_value_objects


async def request_weather_forecast(
    session: aiohttp.ClientSession, station_id: str, params: dict, wfc_raw
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


async def request_simulation(
    session: aiohttp.ClientSession, body: dict, q: asyncio.Queue
):
    """Request simulation by POSTing to `/experiments`."""

    id = body["modelInstanceID"]
    try:
        origin = os.environ["UC1D_SIMAAS_ORIGIN"]
    except KeyError as e:
        logger.critical(f"Required ENVVAR {e} is not set; exciting.")
        sys.exit(EXIT_ENVVAR_MISSING)

    href = f"{origin}/experiments"
    logger.info(f"Requesting simulation of model instance <{id}>...")
    logger.debug(f"POST {href}")

    # Trigger simulation and wait for 201
    async with session.post(href, json=body) as res:
        href_location = res.headers["Location"]
        logger.debug(f"Location: {href_location}")

        # Enqueue link to resource just created
        await q.put((id, href_location))


async def poll_until_done(
    session: aiohttp.ClientSession, q1: asyncio.Queue, q2: asyncio.Queue
):
    """Poll specific experiment until it's done or failed."""

    counter_max = int(os.getenv("UC1D_POLLING_RETRIES", "30"))
    freq = float(os.getenv("UC1D_POLLING_FREQUENCY", "0.1"))

    while True:
        # Retrieve first item from queue
        id, href = await q1.get()

        # Poll status of simulation
        counter = 0
        href_result = None
        while counter < counter_max:
            logger.debug(f"GET {href}")
            async with session.get(href) as res:
                rep = await res.json()
                status = rep["status"]
                logger.debug(f"    {status}")

                if status == "DONE":
                    href_result = rep["linkToResult"]
                    break
                if status == "FAILED":
                    logger.warning("Simulation failed")
                    break

                counter += 1
                await asyncio.sleep(freq)

        # Enqueue link to result
        await q2.put((id, href_result))

        # Indicate that a formerly enqueued task is complete
        q1.task_done()


async def fetch_simulation_result(session: aiohttp.ClientSession, q: asyncio.Queue):
    """Get the simulation result and parse it as dataframe."""

    while True:
        # Retrieve first item from queue
        id, href = await q.get()

        # Get simulation result
        logger.debug(f"GET {href}")
        async with session.get(href) as res:
            rep = await res.json()
            # logger.trace(json.dumps(rep, indent=JSON_DUMPS_INDENT))

        # # Parse as dataframe
        logger.info(f"Result of simulation ~~added to dataframe~~ _dropped_")

        # Indicate that a formerly enqueued task is complete
        q.task_done()


async def weather_forecasts_as_df(start: int, end: int):
    station_id = 10708
    model = "cosmo-d2"
    model_runs = (
        ["03", "09", "15", "21"]
        if model.upper() == "MOSMIX"
        else ["00", "03", "06", "09", "12", "15", "18", "21"]
    )
    quantities = [
        "t_2m",
        "aswdir_s",
        "aswdifd_s",
        "ws_10m",
    ]

    # Get ensemble forecast for weather at a DWD weather station; put data into queue
    q_nwp = []
    async with aiohttp.ClientSession() as session:
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
    ids = []
    frames = []
    for id, body in q_nwp:
        ids.append(str(id))
        frames.append(timeseries_array_as_df(body))

    df = pd.concat(frames, keys=ids)
    logger.trace(f"df\n{df}")

    return df


async def get_simulation_request_bodies():
    period = pendulum.period(
        pendulum.datetime(2018, 11, 17).start_of("day"),
        pendulum.datetime(2018, 11, 17).end_of("day"),
    )  # TODO make configurable
    start = int(period.start.format("x"))
    end = int(period.end.format("x"))
    output_interval = 900

    # Fetch weather forecasts and store in MultiIndex-dataframe
    df = await weather_forecasts_as_df(start, end)

    # outfile = f"/home/moritz/work/projekte/designetz/software/simaas/demo/outfile.csv"
    # df.to_csv(outfile, quoting=csv.QUOTE_NONNUMERIC)
    # df = pd.read_csv(outfile, index_col=[0, 1])

    logger.debug(f"df\n{df}")

    # Construct request bodies from dataframe
    bc = {}
    for model_run in df.index.levels[0]:
        tmp = df.loc[model_run]
        tmp = tmp.interpolate()

        start = tmp.first_valid_index().timestamp() * 1000

        # Create request-body
        request_body = {
            # "modelInstanceID": "c02f1f12-966d-4eab-9f21-dcf265ceac71",
            "modelInstanceID": "29f11d50-f11e-46e7-ba2f-7d69f796a101",
            "simulationParameters": {
                "startTime": start,
                "stopTime": end,
                "outputInterval": output_interval,
            },
            "inputTimeseries": [],
            "startValues": {"plantModel.epochOffset": start / 1000},
        }

        # Update request_body with timeserieses of temp_air, irr_diffuse and irr_direct
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

        bc[model_run] = request_body

    logger.trace(json.dumps(bc, indent=JSON_DUMPS_INDENT))

    return bc


async def ensemble_forecast():
    """Execute ensemble forecast and represent as dataframe."""

    # Assemble data used as input for simulations
    timer_wfc = Timer(
        text="Overall duration for preparing weather forecasts: {:.2f} seconds",
        logger=logger.info,
    )
    with timer_wfc:
        sim_req_bodies = await get_simulation_request_bodies()

    ensemble_runs_total = len(sim_req_bodies)

    # Spawn client session, task queues, producer and workers
    session = aiohttp.ClientSession()
    q_sim = asyncio.Queue()
    q_res = asyncio.Queue()

    requesting = [
        asyncio.create_task(request_simulation(session, body, q_sim))
        for id, body in sim_req_bodies.items()
    ]
    polling = [
        asyncio.create_task(poll_until_done(session, q_sim, q_res))
        for n in range(ensemble_runs_total)
    ]
    fetching = [
        asyncio.create_task(fetch_simulation_result(session, q_res))
        for n in range(ensemble_runs_total)
    ]

    # Await addition of all tasks to the first queue
    await asyncio.gather(*requesting)

    # Wait until the queues are fully processed; then cancel worker tasks&close session
    await q_sim.join()
    await q_res.join()

    for task in polling + fetching:
        task.cancel()

    await session.close()


if __name__ == "__main__":
    timer_overall = Timer(
        text="Overall duration for running ensemble forecast: {:.2f} seconds",
        logger=logger.info,
    )

    with timer_overall:
        asyncio.run(ensemble_forecast())

