import plotly.graph_objects as go
import plotly.io as pio
import base64

from statistics import mean
from domonic.html import div, h3, h2, h1, p, img, table, td, th, tr, thead, tbody
from . import common_components as cc

metric_name_lookup = {
    "mean_squared_error": "Mean squared error",
    "mean_absolute_error": "Mean absolute error",
}


def get_model_overview_page(data):
    return cc.get_model_overview(data)


def get_bar_plot_explanation_image():
    with open("./_score_card/box_plot_explain.png", "rb") as img_file:
        png_base64 = base64.b64encode(img_file.read()).decode("utf-8")
    return div(
        img(_src="data:image/png;base64,{}".format(png_base64), _class="left_img"),
        _class="image_div",
    )


def get_data_explorer_page(data):
    de_heading_left_elms = p(
        "Evaluate your dataset to assess representation of identified subgroups:"
    )

    de_heading_left_elms.append(get_bar_plot_explanation_image())

    de_heading_left_container = div(de_heading_left_elms, _class="left")

    de_main_elems = []

    for c in data:
        de_main_elems.append(h3(c["feature_name"]))
        for i in c["data"]:
            de_main_elems.append(
                p(
                    "For datapoints with {} between {}, {} is the average of the {} between the actual value and the predicted value".format(
                        c["feature_name"],
                        i["label"],
                        round(i[c["primary_metric"]], 1),
                        c["primary_metric"],
                    )
                )
            )
        de_main_elems.append(
            div(
                p(
                    "The distribution of prediction value for the different sets of data points are as follows:"
                ),
                cc.get_de_box_plot_image(c),
                _class="nobreak_div",
            )
        )

    de_main_container = div(de_main_elems, _class="main")

    return str(
        div(
            cc.get_page_divider("Data Explorer"),
            de_heading_left_container,
            de_main_container,
            _class="container",
        )
    )


def get_metric_explanation_text(mname, mvalue):
    text_lookup = {
        "mean_squared_error": "{} is the average of the squared difference between "
        "the actual values and the predicted values.",
        "mean_squared_error": "{} is the average of the squared difference between the actual values and the predicted values.",
        "mean_absolute_error": "{} is the average of the absolute error values.",
        "median_absolute_error": "{} is the median of the absolute error values.",
        "r2_score": "{} is amount of variation in the predicted values that can be explained by the model inputs.",
    }
    return text_lookup[mname].format(round(mvalue, 2))


def get_distributions_plot(data):
    bar_plot_data = {
        "data": [
            {"label": "Predicted<br>Value", "datapoints": data["y_pred"]},
            {"label": "Ground<br>Truth", "datapoints": data["y_test"]},
        ]
    }
    png_base64 = cc.get_box_plot(bar_plot_data)
    return div(
        img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div"
    )


def get_mp_error_histogram_plot(data):
    fig = go.Figure(data=[go.Histogram(y=data["y_error"], nbinsy=10)])
    fig.update_layout(
        paper_bgcolor="rgb(255, 255, 255)",
        plot_bgcolor="rgba(218, 227, 243, 1)",
        margin=dict(l=25, r=25, t=25, b=25),
        width=700,
        height=10 * 70 + 30,
        bargap=0.07,
        xaxis_title_text="Counts",
        yaxis_title_text="Residuals",
    )

    png = pio.to_image(fig)
    png_base64 = base64.b64encode(png).decode("ascii")
    return div(
        img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div"
    )


def get_model_performance_page(data):
    left_metric_elems = [
        p("Observe evidence of model error rates and performance here:")
    ]
    for m in data["metrics"]:
        left_metric_elems.append(h3(metric_name_lookup[m]))
        left_metric_elems.append(p(get_metric_explanation_text(m, data["metrics"][m])))

    left_container = div(left_metric_elems, _class="left")

    main_distributions = div(
        h3("Distributions"),
        get_distributions_plot(data),
        _class="nobreak_div",
    )
    main_histogram = div(
        h3(
            "Histogram of your residuals values "
            "(distance between a predicted value and the observed actual value)"
        ),
        get_mp_error_histogram_plot(data),
        _class="nobreak_div",
    )

    main_container = div(main_distributions, main_histogram, _class="main")
    return str(
        div(
            cc.get_page_divider("Model Performance"),
            left_container,
            main_container,
            _class="container",
        )
    )


def get_cohorts_page(data, metric_config):
    return cc.get_cohorts_page(data, metric_config)


def get_feature_importance_page(data):
    return cc.get_feature_importance_page(data)


def get_causal_page(data):
    return cc.get_causal_page(data)


def get_fairlearn_page(data):
    left_elems = [
        div(
            p(
                "Understand your model’s fairness issues "
                "using group-fairness metrics across sensitive features and cohorts. "
                "Pay particular attention to the subgroups who receive worse treatments "
                "(predictions) by your model."
            )
        )
    ]

    for f in data:
        left_elems.append(h3('Feature "{}"'.format(f)))
        metric_section = []
        for metric_key, metric_details in data[f]["metrics"].items():
            left_elems.append(
                p(
                    '"{}" has the highest {}: {}'.format(
                        metric_details["group_max"][0],
                        metric_key,
                        round(metric_details["group_max"][1], 2),
                    )
                )
            )
            left_elems.append(
                p(
                    '"{}" has the lowest {}: {}'.format(
                        metric_details["group_min"][0],
                        metric_key,
                        round(metric_details["group_min"][1], 2),
                    )
                )
            )
            if metric_details["kind"] == "difference":
                metric_section.append(
                    p(
                        "Maximum difference in {} is {}".format(
                            metric_key,
                            round(
                                metric_details["group_max"][1]
                                - metric_details["group_min"][1],
                                2,
                            ),
                        )
                    )
                )
            elif metric_details["kind"] == "ratio":
                metric_section.append(
                    p(
                        "Minimum ratio of {} is {}".format(
                            metric_key,
                            round(
                                metric_details["group_min"][1]
                                / metric_details["group_max"][1],
                                2,
                            ),
                        )
                    )
                )
        left_elems.append(div(metric_section, _class="nobreak_div"))

    left_container = div(left_elems, _class="left")

    def get_fairness_box_plot(data):
        box_plot_data = {"data": []}
        for c in data:
            box_plot_data["data"].append(
                {
                    "label": c + "<br>" + str(int(100 * data[c]["population"])) + "%",
                    "datapoints": data[c]["y_pred"],
                }
            )

        png_base64 = cc.get_box_plot(box_plot_data)

        return div(
            img(_src="data:image/png;base64,{}".format(png_base64)),
            _class="image_div",
        )

    def get_table_row(heading, data):
        table_row_elems = []
        table_row_elems.append(th(heading, _class="header_cell"))
        for v in data:
            table_row_elems.append(td(v, _class="cell"))
        return tr(table_row_elems, _class="row")

    def get_table(data):
        metric_list = [d for d in data["metrics"]]
        # metric_list = [m for sublist in metric_list for m in sublist]

        horizontal_headings = [
            "Average<br>Prediction",
            "Average<br>Groundtruth",
        ] + [d.replace("_", "<br>") for d in metric_list]
        vertical_headings = list(data["statistics"].keys())

        headings_td = [td(_class="header_cell")] + [
            td(x, _class="header_cell") for x in horizontal_headings
        ]
        headings = thead(tr(headings_td, _class="row"), _class="table-head")

        rows_elems = []
        for vh in vertical_headings:
            row_data = [
                round(mean(data["statistics"][vh]["y_pred"]), 2),
                round(mean(data["statistics"][vh]["y_test"]), 2),
            ]
            for m in metric_list:
                row_data.append(round(data["metrics"][m]["group_metric"][vh], 2))
            rows_elems.append(get_table_row(vh, row_data))

        body = tbody(rows_elems, _class="table-body")

        return table(headings, body, _class="table")

    main_elems = []
    # prediction distribution
    for f in data:
        main_elems.append(
            div(
                h2('Feature "{}"'.format(f)),
                h3("Prediction distribution chart"),
                get_fairness_box_plot(data[f]["statistics"]),
                _class="nobreak_div",
            )
        )

        main_elems.append(
            div(
                h3("Analysis across cohorts"),
                get_table(data[f]),
                _class="nobreak_div",
            )
        )

    main_container = div(main_elems, _class="main")

    return str(
        div(
            cc.get_page_divider("Fairness"),
            left_container,
            main_container,
            _class="container nobreak_div",
        )
    )
