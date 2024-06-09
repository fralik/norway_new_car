# PyShiny example

This PyShiny application is designed to gather insights about used cars sold in Norway. It provides a user-friendly interface for users to explore and analyze data related to used car sales in Norway. The app utilizes data from the most popular Norwegian website to provide valuable information such as average prices, popular car models, mileage trends, e.t.c.

The app collects data about cars in a joblib file.

## Usage

1. Grab a copy of the source code
2. Prepare python virtual environment. I used Python 3.11 in my development, but other versions should work fine.
Run:
```
pip install -r requirements.txt
cd src
shiny run app.py --reload
```

Since this is a very one-man project, no deployment is proposed. The code is intended to be run locally.

## Development

For development, it is recommended to:
```
pip install -r requirements-dev.txt
pre-commit install
```
