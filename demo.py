#!/usr/bin/env python3
# -*- coding: utf8 -*-

# SPDX-FileCopyrightText: 2021 UdS AES <https://www.uni-saarland.de/lehrstuhl/frey.html>
# SPDX-License-Identifier: MIT


"""Demonstrate performance of SIMaaS-implementation using async IO."""

import asyncio
import json
import os
import sys

import aiohttp
from codetiming import Timer
from loguru import logger

JSON_DUMPS_INDENT = 2  # intendation width for stringified JSON in spaces

# Configure logging
log_level = os.getenv("UC1D_LOG_LEVEL", "INFO")
logger.remove()
logger.add(sys.stdout, level=log_level, diagnose=True, backtrace=False, enqueue=True)


async def request_simulation(
    session: aiohttp.ClientSession, body: dict, q: asyncio.Queue
):
    """Request simulation by POSTing to `/experiments`."""

    id = body["modelInstanceID"]
    origin = os.getenv("UC1D_SIMAAS_ORIGIN")
    href = f"{origin}/experiments"
    logger.trace(f"POST {href}")

    # Trigger simulation and wait for 201
    async with session.post(href, json=body) as res:
        href_location = res.headers["Location"]
        logger.debug(f"Location: {href_location}")

        logger.info(f"Requested simulation of model instance <{id}>")

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
            logger.trace(f"GET {href}")
            async with session.get(href) as res:
                rep = await res.json()
                status = rep["status"]
                logger.trace(f"    {status}")

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
        logger.trace(f"GET {href}")
        async with session.get(href) as res:
            rep = await res.json()
            # logger.trace(json.dumps(rep, indent=JSON_DUMPS_INDENT))

        # # Parse as dataframe
        # logger.info(f"Result of simulation added to dataframe")

        # Indicate that a formerly enqueued task is complete
        q.task_done()


def get_simulation_request_bodies():
    eg_body = {
        "modelInstanceID": "c02f1f12-966d-4eab-9f21-dcf265ceac71",
        "simulationParameters": {
            "startTime": 1542412800000,
            "stopTime": 1542499199999,
            "outputInterval": 3600,
        },
        "inputTimeseries": [
            {
                "label": "temperature",
                "unit": "K",
                "timeseries": [
                    {"timestamp": 1542412800000, "value": 274.6336669921875},
                    {"timestamp": 1542416400000, "value": 274.4828796386719},
                    {"timestamp": 1542420000000, "value": 274.01922607421875},
                    {"timestamp": 1542423600000, "value": 273.8811340332031},
                    {"timestamp": 1542427200000, "value": 273.7162628173828},
                    {"timestamp": 1542430800000, "value": 273.4994201660156},
                    {"timestamp": 1542434400000, "value": 273.3451385498047},
                    {"timestamp": 1542438000000, "value": 273.2210388183594},
                    {"timestamp": 1542441600000, "value": 273.4631805419922},
                    {"timestamp": 1542445200000, "value": 274.3274230957031},
                    {"timestamp": 1542448800000, "value": 275.4784393310547},
                    {"timestamp": 1542452400000, "value": 276.50999450683594},
                    {"timestamp": 1542456000000, "value": 277.6051940917969},
                    {"timestamp": 1542459600000, "value": 278.2355194091797},
                    {"timestamp": 1542463200000, "value": 278.0879669189453},
                    {"timestamp": 1542466800000, "value": 277.3506317138672},
                    {"timestamp": 1542470400000, "value": 276.5809631347656},
                    {"timestamp": 1542474000000, "value": 275.99607849121094},
                    {"timestamp": 1542477600000, "value": 275.36585998535156},
                    {"timestamp": 1542481200000, "value": 274.8233642578125},
                    {"timestamp": 1542484800000, "value": 274.35382080078125},
                    {"timestamp": 1542488400000, "value": 273.8891143798828},
                    {"timestamp": 1542492000000, "value": 273.6193542480469},
                    {"timestamp": 1542495600000, "value": 273.1964416503906},
                    {"timestamp": 1542499200000, "value": 272.87730407714844},
                ],
            },
            {
                "label": "directHorizontalIrradiance",
                "unit": "W/m²",
                "timeseries": [
                    {"timestamp": 1542412800000, "value": 0},
                    {"timestamp": 1542416400000, "value": 0},
                    {"timestamp": 1542420000000, "value": 0},
                    {"timestamp": 1542423600000, "value": 0},
                    {"timestamp": 1542427200000, "value": 0},
                    {"timestamp": 1542430800000, "value": 0},
                    {"timestamp": 1542434400000, "value": 0},
                    {"timestamp": 1542438000000, "value": 0.01708984375},
                    {"timestamp": 1542441600000, "value": 32.473876953125},
                    {"timestamp": 1542445200000, "value": 106.16259765625},
                    {"timestamp": 1542448800000, "value": 172.1171875},
                    {"timestamp": 1542452400000, "value": 207.6816406250005},
                    {"timestamp": 1542456000000, "value": 203.74609375089045},
                    {"timestamp": 1542459600000, "value": 161.24023440730758},
                    {"timestamp": 1542463200000, "value": 92.67578142064986},
                    {"timestamp": 1542466800000, "value": 21.17968753655803},
                    {"timestamp": 1542470400000, "value": 0.013671875010348028},
                    {"timestamp": 1542474000000, "value": -0.011718749990169372},
                    {"timestamp": 1542477600000, "value": 0.027343750008278423},
                    {"timestamp": 1542481200000, "value": 0.003906250007761022},
                    {"timestamp": 1542484800000, "value": -0.013671874993791184},
                    {"timestamp": 1542488400000, "value": -0.046874999994825986},
                    {"timestamp": 1542492000000, "value": -0.07421874999482599},
                    {"timestamp": 1542495600000, "value": -0.041015624994825986},
                    {"timestamp": 1542499200000, "value": -0.041015624995343385},
                ],
            },
            {
                "label": "diffuseHorizontalIrradiance",
                "unit": "W/m²",
                "timeseries": [
                    {"timestamp": 1542412800000, "value": 0},
                    {"timestamp": 1542416400000, "value": 0},
                    {"timestamp": 1542420000000, "value": 0},
                    {"timestamp": 1542423600000, "value": 0},
                    {"timestamp": 1542427200000, "value": 0},
                    {"timestamp": 1542430800000, "value": 0},
                    {"timestamp": 1542434400000, "value": 0},
                    {"timestamp": 1542438000000, "value": 3.03173828125},
                    {"timestamp": 1542441600000, "value": 51.814697265625},
                    {"timestamp": 1542445200000, "value": 80.36165171861649},
                    {"timestamp": 1542448800000, "value": 95.63700693845749},
                    {"timestamp": 1542452400000, "value": 102.13556969165802},
                    {"timestamp": 1542456000000, "value": 101.35517811775208},
                    {"timestamp": 1542459600000, "value": 93.04704689979553},
                    {"timestamp": 1542463200000, "value": 74.65015459060669},
                    {"timestamp": 1542466800000, "value": 43.18778944015503},
                    {"timestamp": 1542470400000, "value": -0.00856328010559082},
                    {"timestamp": 1542474000000, "value": -0.06733894348144531},
                    {"timestamp": 1542477600000, "value": 0.03996133804321289},
                    {"timestamp": 1542481200000, "value": -0.05777382850646973},
                    {"timestamp": 1542484800000, "value": 0.086669921875},
                    {"timestamp": 1542488400000, "value": 0.07489943504333496},
                    {"timestamp": 1542492000000, "value": -0.03650379180908203},
                    {"timestamp": 1542495600000, "value": 0.022024869918823242},
                    {"timestamp": 1542499200000, "value": 0.025892019271850586},
                ],
            },
        ],
    }
    bc = {
        # "29f11d50-f11e-46e7-ba2f-7d69f796a101": {},
        # "723ca3b3-2a02-4356-b5b7-7b52a1d65e4b": {},
        # "932168ed-8c4c-46e6-9eb9-25e2dec8f6c9": {},
        # "bc5bdfd1-c537-42b1-b44d-d200ae44bfbd": {},
        "c02f1f12-966d-4eab-9f21-dcf265ceac71": {},
        # "c17c78f3-5100-41a8-a189-cc9c2df0e3f9": {},
        # "c7d7cf43-63f2-450c-833d-69feaca5e05c": {},
    }

    for key, item in bc.items():
        tmp = eg_body
        # tmp["modelInstanceID"] = key
        item.update(tmp)

    logger.trace(json.dumps(bc, indent=JSON_DUMPS_INDENT))

    return bc


async def ensemble_forecast():
    """Execute ensemble forecast and represent as dataframe."""

    # Assemble data used as input for simulations
    sim_req_bodies = get_simulation_request_bodies()
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

