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


async def request_simulation(body: dict, q: asyncio.Queue):
    """Request simulation by POSTing to `/experiments`."""

    id = body["modelInstanceID"]
    origin = os.getenv("UC1D_SIMAAS_ORIGIN")
    href = f"{origin}/experiments"
    logger.trace(f"POST {href}")

    async with aiohttp.ClientSession() as session:
        # Trigger simulation and wait for 201
        async with session.post(href, json=body) as res:
            href_location = res.headers["Location"]
            logger.debug(f"Location: {href_location}")

            logger.info(f"Requested simulation of model instance <{id}>")

            # Enqueue link to resource just created
            await q.put((id, href_location))


async def poll_until_done(q1: asyncio.Queue, q2: asyncio.Queue):
    """Poll specific experiment until it's done or failed."""

    counter_max = int(os.getenv("UC1D_POLLING_RETRIES", "30"))
    freq = float(os.getenv("UC1D_POLLING_FREQUENCY", "0.1"))

    while True:
        # Retrieve first item from queue
        id, href = await q1.get()

        # Poll status of simulation
        async with aiohttp.ClientSession() as session:
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


async def fetch_simulation_result(q: asyncio.Queue):
    """Get the simulation result and parse it as dataframe."""

    while True:
        # Retrieve first item from queue
        id, href = await q.get()

        async with aiohttp.ClientSession() as session:
            # Get simulation result
            logger.trace(f"GET {href}")
            async with session.get(href) as res:
                rep = await res.json()
                # logger.trace(json.dumps(rep, indent=JSON_DUMPS_INDENT))

        # # Parse as dataframe
        # logger.info(f"Result of simulation added to dataframe")

        # Indicate that a formerly enqueued task is complete
        q.task_done()


async def ensemble_forecast():
    """Execute ensemble forecast and represent as dataframe."""

    # Assemble data used as input for simulations
    bc = {
        0: "a",
        1: "b",
    }
    ensemble_runs_total = len(bc)

    # Spawn task queues, producer and workers
    q_sim = asyncio.Queue()
    q_res = asyncio.Queue()

    requesting = [
        asyncio.create_task(request_simulation(body, q_sim)) for id, body in bc.items()
    ]
    polling = [
        asyncio.create_task(poll_until_done(q_sim, q_res))
        for n in range(ensemble_runs_total)
    ]
    fetching = [
        asyncio.create_task(fetch_simulation_result(q_res))
        for n in range(ensemble_runs_total)
    ]

    # Await addition of all tasks to the first queue
    await asyncio.gather(*requesting)

    # Wait until the queues are fully processed; then cancel worker tasks
    await q_sim.join()
    await q_res.join()

    for task in polling + fetching:
        task.cancel()


if __name__ == "__main__":
    timer_overall = Timer(
        text="Overall duration for running ensemble forecast: {:.2f} seconds",
        logger=logger.info,
    )

    with timer_overall:
        asyncio.run(ensemble_forecast())

