"""
Module for scraping car ads from finn.no. We create a data frame and store it in a joblib file.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import joblib
import pandas as pd
import requests
from loguru import logger


def _update_url_page_param(url: str, new_page: int) -> str:
    # Parse the URL into components
    url_parts = urlparse(url)

    # Parse the URL's query parameters
    query_params = parse_qs(url_parts.query)

    # Update the 'page' parameter
    query_params["page"] = [str(new_page)]

    # Construct the new URL
    new_query_string = urlencode(query_params, doseq=True)
    new_url_parts = url_parts._replace(query=new_query_string)
    new_url = urlunparse(new_url_parts)

    return str(new_url)


def _scrap_cars(year_from: int = 2016, year_to: int = 2016):
    # VW: passat, passat alltrack, tiguan
    vw_passat_tiguan = "model=1.817.2000209&model=1.817.1437&model=1.817.8329"
    models = {
        "passat": "1.817.1437",
        # "passat_alltrack": "1.817.2000209",
        "tiguan": "1.817.8329",
        "octavia": "1.808.1355",
        # "octavia_rs": "1.808.2000335",
        # "octavia_scout": "1.808.2000336",
        "skoda_superb": "1.808.7532",
        "golf": "1.817.1433",
        "golf_alltrack": "1.817.2000331",
    }

    ""
    # sales_form=1 - used cars for sale
    # fuel=1,2,6 - benzin, diesel, hybrid
    # car_equipment=23 - hengerfeste
    # search_url_base = "https://www.finn.no/api/search-qf?searchkey=SEARCH_ID_CAR_USED&car_equipment=23&fuel=1&fuel=2&fuel=6&geoLocationName=Landingsveien+138%2C+Oslo&lat=59.95151&lon=10.64995&radius=100000&sales_form=1&q=&year_to={year_to}&year_from={year_from}&vertical=car&{model}"
    search_url_base = "https://www.finn.no/api/search-qf?searchkey=SEARCH_ID_CAR_USED&car_equipment=23&fuel=1&fuel=2&fuel=6&geoLocationName=Landingsveien+138%2C+Oslo&lat=59.958358012580554&lon=10.800518430769444&radius=100000&sales_form=1&q=&year_to={year_to}&year_from={year_from}&vertical=car&{model}"
    db = []

    for key, item in models.items():
        logger.debug(f"Scraping {key}")
        model_url_param = f"model={item}"
        search_url = search_url_base.format(year_to=year_to, year_from=year_from, model=model_url_param)

        while True:
            resp = requests.get(search_url, timeout=3)
            if resp.status_code != 200:
                logger.error(f"Error: {resp.status_code} from {search_url}, {resp.text}")
                break
            results_json = resp.json()
            for d in results_json["docs"]:
                timestamp_sec = d["timestamp"] / 1000  # convert to seconds
                py_timestamp = datetime.fromtimestamp(timestamp_sec)

                db.append(
                    {
                        "id": d["id"],
                        "model_id": item,
                        "model_name": key,
                        "lat": d["coordinates"]["lat"],
                        "lon": d["coordinates"]["lon"],
                        "url": d["canonical_url"],
                        "price": d["price"]["amount"],
                        "year": d["year"],
                        "mileage": d["mileage"],
                        "dealer_segment": d["dealer_segment"],
                        "heading": d["heading"],
                        "sold": "sold" in d["flags"],
                        "created_at": py_timestamp,
                    }
                )
            current_page = int(results_json["metadata"]["paging"]["current"])
            last_page = int(results_json["metadata"]["paging"]["last"])
            if current_page > last_page:
                break

            # prepare new url and continue
            search_url = _update_url_page_param(search_url, current_page + 1)
    return db


def scrap(year_from: int = 2016, year_to: int = 2016, output_file: str = "cars.joblib"):
    if Path(output_file).exists():
        db = joblib.load(output_file)
        df = db["data"]
    else:
        df = pd.DataFrame()

    try:
        db_scrapped = _scrap_cars(year_from, year_to)
        db_scrapped_map = {item["id"]: item for item in db_scrapped}

        # add new items to df
        df = pd.concat([df, pd.DataFrame(db_scrapped_map.values())]).drop_duplicates(subset=["id"])

        # update all sold flags in df for items in db_scrapped_map
        df = df.assign(
            sold=df.apply(
                lambda row: db_scrapped_map[row["id"]]["sold"] if row["id"] in db_scrapped_map else True, axis=1
            )
        )
        timestamp = datetime.now()

        joblib.dump({"data": df, "last_updated": timestamp}, output_file)

    except Exception as e:
        logger.debug(e)
