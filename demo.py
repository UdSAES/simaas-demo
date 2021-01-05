#!/usr/bin/env python3
# -*- coding: utf8 -*-

# SPDX-FileCopyrightText: 2021 UdS AES <https://www.uni-saarland.de/lehrstuhl/frey.html>
# SPDX-License-Identifier: MIT


"""Demonstrate performance of SIMaaS-implementation using async IO."""

import asyncio
import os
import sys

from codetiming import Timer
from loguru import logger


# Configure logging
log_level = os.getenv("UC1D_LOG_LEVEL", "INFO")
logger.remove()
logger.add(sys.stdout, level=log_level, diagnose=True, backtrace=False)


async def request_simulations(data: dict, s: aiohttp.ClientSession, q: asyncio.Queue):
    """Request simulation by POSTing to `/experiments`."""

    for id, body in data.items():
        # Trigger simulation and wait for 201
        # await asyncio.sleep(0.5)
        href_location = f"href_loc_{id}"

        logger.info(f"Requested simulation for ensemble run <{id}>")

        # Enqueue link to resource just created
        await q.put((id, href_location))


async def poll_until_done(s: aiohttp.ClientSession, q1: asyncio.Queue, q2: asyncio.Queue):
    """Poll specific experiment until it's done or failed."""

    while True:
        # Retrieve first item from queue
        id, href = await q1.get()

        # Poll status of simulation
        await asyncio.sleep(1)
        status = "done"

        logger.debug(f"Polled status of simulation for ensemble run <{id}>")
        logger.info(f"Simulation <{id}> finished with status {status}")

        # Enqueue link to result
        href_result = "qwer"
        await q2.put((id, href_result))

        # Indicate that a formerly enqueued task is complete
        q1.task_done()


async def fetch_simulation_result(s: aiohttp.ClientSession, q: asyncio.Queue):
    """Get the simulation result and parse it as dataframe."""

    while True:
        # Retrieve first item from queue
        id, href = await q.get()

        # Get simulation result
        # await asyncio.sleep(0.5)

        # Parse as dataframe
        logger.info(f"Result of simulation <{id}> added to dataframe")

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

    requesting = [asyncio.create_task(request_simulations(bc, q_sim))]
    polling = [
        asyncio.create_task(poll_until_done(q_sim, q_res))
        for n in range(ensemble_runs_total)
    ]
    fetching = [
        asyncio.create_task(fetch_simulation_result(q_res))
        for n in range(ensemble_runs_total)
    ]

    # Await addition of all tasks to the first queue
    await asyncio.gather(*requesting, return_exceptions=True)

    # Wait until the queues are fully processed; then cancel worker tasks
    await q_sim.join()
    await q_res.join()

    for task in polling + fetching:
        task.cancel()


if __name__ == "__main__":
    timer_overall = Timer(text="Overall duration for running ensemble forecast: {:.2f} seconds", logger=logger.info)

    with timer_overall:
        asyncio.run(ensemble_forecast())

