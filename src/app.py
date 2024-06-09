import functools
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import plotly
import plotly.graph_objects as go
from loguru import logger
from shinywidgets import output_widget, render_plotly

import scrap_utils
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui

# Constants, could become app settings in the future
OK_MILEAGE = 12000
HIGH_MILEAGE = 15000
AGE_REF_YEAR = datetime.now().year


def logger_wraps(*, entry=True, exit=True, level="DEBUG"):
    """Decorator to wrap a function with log entries.

    Use as @logger_wraps() or @logger_wraps(entry=False) or @logger_wraps(exit=False)
    """

    def wrapper(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            logger_ = logger.opt(depth=1)
            if entry:
                logger_.log(level, "Entering '{}'", name)
            result = func(*args, **kwargs)
            if exit:
                logger_.log(level, "Exiting '{}'", name)
            return result

        return wrapped

    return wrapper


def make_mileage_box(car_year: int, reference_year: int = AGE_REF_YEAR):
    """Create a value box with low/high mileage range for a given car year."""
    age = reference_year - car_year
    high_mileage = age * HIGH_MILEAGE
    low_mileage = age * OK_MILEAGE
    return ui.value_box(
        title=f"{car_year}",
        value="{:,} - {:,} km".format(low_mileage, high_mileage).replace(",", " "),
        theme="primary",
    )


app_ui = ui.page_sidebar(
    ui.sidebar(
        # Sidebar contains a slider for cars year range, button to fetch cars from finn.no and DB status information
        ui.input_slider(
            "slider_year_range", "Car year range", min=2010, max=AGE_REF_YEAR - 1, value=[2014, 2022], step=1, sep=""
        ),
        ui.input_action_button("inp_btn_scrap", "Fetch cars", class_="btn-primary"),
        ui.output_ui(id="ui_db_info"),
    ),
    # Make 3 boxes for low/high mileage range for each year
    ui.layout_columns(*[make_mileage_box(year) for year in range(2014, 2017)]),
    # Show Top 3 car models by the amount of ads
    ui.output_ui(id="out_top_cars"),
    ui.layout_columns(
        ui.card(
            ui.card_header("Mileage Status by Year"),
            ui.output_plot("plot_mileage_status_by_year"),
            full_screen=True,
        ),
        ui.card(
            ui.card_header("Cars for sale per model (yearly)"),
            ui.output_plot("plot_num_for_sale_per_year"),
            full_screen=True,
        ),
        # style="min-height: 800px;",
    ),
    ui.layout_columns(
        ui.card(
            ui.card_header("Price vs Mileage for selected model"),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select("inp_select_model", "Select model", choices=[]),
                    ui.input_select("inp_select_year", "Select year", choices=[], multiple=True),
                ),
                output_widget("plot_price_mileage_dependency"),
            ),
            full_screen=True,
        ),
    ),
    ui.card(
        ui.card_header("Price over years and mileage"),
        ui.layout_sidebar(
            ui.sidebar(
                ui.input_select("inp_select_model_price_stats", "Select model", choices=[]),
            ),
            ui.output_data_frame("df_price_stats"),
        ),
    ),
    ui.tags.script(src="app.js", type="module"),
    title="Cars from Finn.no",
    lang="en",
)


@logger_wraps()
def get_price_mileage_dependency_figure(
    df: pd.DataFrame, model_name: str, years: Optional[list[int]] = None
) -> go.FigureWidget:
    if not years:
        years = [2014, 2015]
    if df.empty:
        return go.FigureWidget()

    # Filter the data for the selected model_name
    df_display = df[(df["model_name"] == model_name) & (df["year"].isin(years))].copy().sort_values(["mileage", "year"])

    # mileage_status can be -1, 0, 1. Make it 0, 1
    df_display.loc[df_display["mileage"] < df_display["upper_bound"], "mileage_status"] = 0

    # Create a color map
    colors = plotly.colors.qualitative.Plotly
    color_map = {year: color for year, color in zip(sorted(df["year"].unique()), colors)}

    fig = go.FigureWidget()

    with fig.batch_update():
        # clear existing figure
        fig.data = []

        for year in years:
            df_year = df_display[df_display["year"] == year]
            if df_year.empty:
                continue

            for status, line_dash, label in [(0, "solid", "OK"), (1, "dash", "HIGH")]:
                df_year_status = df_year[df_year["mileage_status"] == status]
                if df_year_status.empty:
                    continue

                color = color_map[year]
                trace = go.Scatter(
                    x=df_year_status["mileage"],
                    y=df_year_status["price"],
                    mode="lines+markers",
                    name=f"{year} ({label} mileage)",
                    hoverinfo="text",
                    text=df_year_status["id"],
                    line=dict(dash=line_dash, color=color),
                    hovertemplate="<b>ID:</b> %{text}<br>%{customdata[0]}<br><b>Km: </b>%{customdata[1]}<br><b>Year: </b>%{customdata[2]}<br><b>Price: </b>%{customdata[3]}",
                    customdata=df_year_status[["heading", "mileage", "year", "price", "id", "url"]].values,
                )
                fig.add_trace(trace)

    fig.update_layout(title=f"Price vs Mileage for {model_name}", xaxis_title="Mileage, km", yaxis_title="Price, NOK")
    return fig


def server(input: Inputs, output: Outputs, session: Session):
    # reactive value to indicate if the db was updated
    db_updated = reactive.Value(False)
    data_df = reactive.Value(pd.DataFrame())
    db_last_updated = reactive.Value("")

    @reactive.effect
    @reactive.event(input.inp_btn_scrap)
    async def on_click_fetch_cars():
        get_cars_from_server()
        db_updated.set(True)

    @reactive.effect
    @reactive.event(data_df)
    def event_update_model_selector():
        df = data_df.get()
        if df.empty:
            logger.debug("DB is empty")
            return

        model_names = df.model_name.unique().tolist()
        update_model_selector(model_names)

    def update_model_selector(model_names: list[str]):
        # check if we have any cars selected already
        selected_model = input.inp_select_model()
        if selected_model not in model_names:
            selected_model = None
        ui.update_select("inp_select_model", choices=model_names, selected=selected_model)

        selected_model = input.inp_select_model_price_stats()
        if selected_model not in model_names:
            selected_model = None
        ui.update_select("inp_select_model_price_stats", choices=model_names, selected=selected_model)

    @reactive.effect
    @reactive.event(input.inp_select_model)
    @logger_wraps()
    def on_model_selected():
        model_name = input.inp_select_model()
        df = data_df.get()
        if df.empty:
            return

        previously_selected = [int(x) for x in input.inp_select_year()]
        years = sorted(df[df["model_name"] == model_name].year.unique().tolist())
        # get overlap between previously selected and available years
        selected_years = (list(set(years) & set(previously_selected))) or years
        ui.update_select("inp_select_year", choices=years, selected=selected_years)

    @reactive.effect
    @reactive.event(input.year_range)
    def year_range_updated():
        # This fires when we receive a message from the client
        year_range = input.year_range()
        ui.update_slider("slider_year_range", value=year_range)

    @reactive.effect
    @reactive.event(input.slider_year_range)
    async def slider_year_range_changed():
        # This fires when the slider is changed via UI
        year_range = input.slider_year_range()
        await session.send_custom_message("msg_saveYearRange", year_range)

    @reactive.calc
    def get_cars_from_server():
        logger.debug(f"In get_cars_from_server from server: {input.year_range()}")
        try:
            req(input.inp_btn_scrap)

            start_year, end_year = input.year_range()
            with ui.Progress(min=1, max=10) as p:
                p.set(message="Fetching cars", detail="Please wait", value=5)

                scrap_utils.scrap(start_year, end_year)
        except Exception as e:
            logger.error(e)
            ui.notification_show(f"Error: {e}", duration=15, type="warning")

    @reactive.effect
    @logger_wraps()
    def get_cars_from_db() -> tuple[pd.DataFrame, datetime]:
        db_filename = "cars.joblib"

        # ensure we have cars
        db_value = db_updated()
        db_exists = Path(db_filename).exists()
        if not db_value and not db_exists:
            logger.debug("No cars in db")
            return pd.DataFrame(), datetime.now()

        try:
            db = joblib.load(db_filename)
            df = db["data"].copy()

            # Leave only unsold cars
            df = df[~df.sold]

            # Add mileage status column
            df["car_age"] = AGE_REF_YEAR - df["year"]
            df["lower_bound"] = df["car_age"] * OK_MILEAGE
            df["upper_bound"] = df["car_age"] * HIGH_MILEAGE

            # -1, 0, 1 == low, OK, high
            df.loc[df["mileage"] < df["lower_bound"], "mileage_status"] = -1
            df.loc[(df["mileage"] >= df["lower_bound"]) & (df["mileage"] <= df["upper_bound"]), "mileage_status"] = 0
            df.loc[df["mileage"] > df["upper_bound"], "mileage_status"] = 1

            # update_model_selector(df.model_name.unique().tolist())
            data_df.set(df)
            db_last_updated.set(db["last_updated"].strftime("%Y-%m-%d %H:%M:%S"))

            logger.debug(f"Returning {df.shape[0]} rows")
            return df, db["last_updated"]
        except Exception as e:
            logger.debug(f"Error in get_cars_from_db: {e}")
            return pd.DataFrame(), datetime.now()

    @render.ui
    def ui_db_info():
        df = data_df.get()
        last_updated = db_last_updated.get()
        if df.empty:
            return ui.markdown("No cars in the database")

        return ui.markdown(
            f"""Cars in the DB: {df.shape[0]}
            Updated: {last_updated}
            """
        )

    @output
    @render_plotly
    @logger_wraps()
    def plot_price_mileage_dependency():
        df = data_df.get()
        if df.empty:
            return
        years = [int(x) for x in input.inp_select_year()]
        fig = get_price_mileage_dependency_figure(df, input.inp_select_model(), years)

        # sub-function to handle click events on data points, we want to open Finn.no on click
        def click_handler(trace, points, selector):
            if points.point_inds:
                ind = points.point_inds[0]
                point = trace.customdata[ind]

                # url is the last element in the customdata
                webbrowser.open_new_tab(point[-1])

        # add click handler to all traces
        for idx, _ in enumerate(fig.data):
            fig.data[idx].on_click(click_handler)

        return fig

    @render.ui
    @logger_wraps()
    def out_top_cars():
        """Render top 3 car models by the amount of ads."""
        df = data_df.get()
        if df.empty:
            return ui.layout_columns(ui.value_box(title="No cars in the database", value="", theme="danger"))

        top_models = df.model_name.value_counts()[:3].reset_index()
        columns = []
        for row in top_models.itertuples():
            columns.append(ui.value_box(title=row.model_name, value=row.count, theme="primary"))

        return ui.layout_columns(*columns)

    @render.plot(alt="plot_mileage_status_by_year")
    @logger_wraps()
    def plot_mileage_status_by_year():
        fig, ax = plt.subplots()

        df = data_df.get()
        if df.empty:
            return fig

        df.groupby("year")["mileage_status"].value_counts().unstack().plot(
            kind="bar",
            stacked=True,
            ax=ax,
            xlabel="Year",
            ylabel="Count",
            title="Mileage Status by Year",
            # legend=['Low Mileage', 'Medium Mileage', 'High Mileage']
        )
        plt.legend(["Low Mileage", "Medium Mileage", "High Mileage"])
        return fig

    @render.plot(alt="plot_num_for_sale_per_year")
    def plot_num_for_sale_per_year():
        fig, ax = plt.subplots()
        df = data_df.get()
        if df.empty:
            return fig

        # size returns number of elements per group
        df.groupby(["year", "model_name"]).size().unstack().plot(
            kind="bar",
            stacked=True,
            ax=ax,
            xlabel="Year",
            ylabel="Count",
            title="Cars for sale per model (yearly)",
        )
        return fig

    @render.data_frame
    def df_price_stats():
        df = data_df.get()
        if df.empty:
            return render.DataGrid(df, selection_mode="row")

        price_stats = (
            df.groupby(["model_name", "mileage_status", "year"])["price"]
            .agg(["count", "min", "max", "mean", "median"])
            .reset_index()
        )
        price_stats = price_stats[price_stats.model_name == input.inp_select_model_price_stats()]
        price_stats = price_stats.drop(columns=["model_name"])
        # map mile_status to string
        mileage_status_map = {-1: "Low Mileage", 0: "Medium Mileage", 1: "High Mileage"}
        price_stats["mileage_status"] = price_stats["mileage_status"].map(mileage_status_map)

        # round mean and median
        price_stats["mean"] = price_stats["mean"].round(0)
        price_stats["median"] = price_stats["median"].round(0)

        return render.DataGrid(price_stats, selection_mode="row")


app = App(app_ui, server, static_assets=Path(__file__).parent / "static")
