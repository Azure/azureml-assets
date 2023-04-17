# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import base64

import pdfkit
import plotly.graph_objects as go
import plotly.io as pio
from domonic.html import (a, div, h1, h3, img, li, p, span, table, tbody, td,
                          thead, tr, ul)


def get_full_html(htmlbody):
    return "<!DOCTYPE html><html><body><head>{}</head>{}</body></html>".format(
        get_css(), htmlbody
    )


def get_css():
    return """<style>
        * {
          font-family: Ubuntu;
        }

        .header {
          /* border: 2px solid green; */
          /*           background-color: #ccf; */
        }

        .container {
          break-inside: avoid !important;
          page-break-inside: avoid !important;
          position: relative;
          min-height: 80%;
          overflow: hidden;
        }

        .left {
          float: left;
          width: 3in;
          padding-bottom: 9999px;
          margin-bottom: -9999px;
        }

        .main {
          position: relative;
          margin-left: 3.05in;
          padding-bottom: 9999px;
          margin-bottom: -9999px;
        }

        .left_model_overview {
          float: left;
          width: 4.2in;
          padding-bottom: 9999px;
          margin-bottom: -9999px;
        }

        .main_model_overview {
          position: relative;
          margin-left: 4.25in;
          padding-bottom: 9999px;
          margin-bottom: -9999px;
        }

        #footer {
          background-color: #fcc;
        }

        .box {
          width: 5in;
          height: auto;
        }

        img {
          width: 5in;
          height: auto;
        }

        .left_img {
          width: 3in;
          height: auto;
        }

        .image_div {
          break-inside: avoid;
        }

        .nobreak_div {
          break-inside: avoid !important;
          page-break-inside: avoid !important;
        }

        .nobreak_div_padding {
          content: "";
          display: block;
          height: 100px;
          margin-bottom: -100px;
          break-inside: avoid !important;
          page-break-inside: avoid !important;
        }

        .cell {
          border-collapse: collapse;
          border: 0.5px solid rgba(199, 199, 199, 1);
          padding: 5px 10px;
        }

        .header_cell {
          border-collapse: collapse;
          border: 0px;
          padding: 5px 10px;
        }

        .table {
          border-collapse: collapse;
          border-style: hidden;
        }

      </style>"""


def get_page_divider(text):
    elem = div(
        _style="height:36px; margin-top: 10px; page-break-after: avoid;",
        _class="nobreak_div",
    ).append(
        div(
            _style="width: 100%; height: 12px; border-bottom: 1px solid #605E5C;"
        ).append(
            span(
                text,
                _style="font-size: 14px; background-color: #605E5C; "
                "padding: 0 10px; border-radius: 18px; color: #ffffff",
            )
        )
    )
    return elem


def get_de_image(data):
    png_base64 = get_de_bar_plot(data)
    return str(
        div(img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div")
    )


def get_cohorts_performance_image(data, m, get_bar_plot_func):
    png_base64 = get_bar_plot_func(data, m)
    return div(
        img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div"
    )


def get_fi_image(data):
    png_base64 = get_fi_bar_plot(data)
    return div(
        img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div"
    )


def get_cohorts_performance_container(data, m, get_bar_plot_func, message):
    cohort_left_heading = h3("{}: {}".format(message["left_heading"], m))
    cohorts_list = ul()
    for c in data:
        cohorts_list.append(li("{}: {}".format(c["short_label"], c["label"])))

    container = div(cohort_left_heading, cohorts_list, _class="nobreak_div")

    cp_left_container = div(container, _class="left")

    cp_main_heading = h3("{}: {}".format(message["main_heading"], m))
    cp_main_image = get_cohorts_performance_image(data, m, get_bar_plot_func)

    container = div(cp_main_heading, cp_main_image, _class="nobreak_div")

    cp_main_container = div(container, _class="main")

    return str(div(cp_left_container, cp_main_container, _class="container"))


def get_feature_importance_page(data):
    fi_left_heading = p(
        "Understand factors that have impacted your model predictions the most. "
        "These are factors that may account for performance levels and differences."
    )
    feature_list = ul()
    for k, v in data.items():
        feature_list.append(li("{}: {}".format(v["short_label"], k)))

    container = div(fi_left_heading, feature_list, _class="nobreak_div")

    fi_left_container = div(container, _class="left")

    heading_main = div(
        h3("Feature Importance"),
        p("Shown below is the mean of SHAP value of the most important features:"),
        get_fi_image(data),
        _class="main",
    )
    return div(
        get_page_divider("Feature Importance (Explainability)"),
        fi_left_container,
        heading_main,
        _class="container",
    )


def get_fi_bar_plot(data):
    y_data = [v["short_label"] for k, v in data.items()]
    x_data = [v["value"] for k, v in data.items()]
    max_x = max(x_data)
    x_range = [0, max_x]
    x_data = [[x, max_x - x] for x in x_data]

    # tickvals = [0.0, 0.25, 0.5, 0.75, 1.0]
    # ticktext = [0.0, 0.25, 0.5, 0.75, 1.0]
    tickappend = ""

    def scientific_formatter(x):
        return "{:.2e}".format(x)

    def rounding_formatter(x):
        return str(round(x, 1))

    if any(
        (i >= 10000 or i <= 0.01 for i in [x for sublist in x_data for x in sublist])
    ):
        text_formatter = scientific_formatter
    else:
        text_formatter = rounding_formatter

    return get_bar_plot(
        list(reversed(y_data)),
        list(reversed(x_data)),
        tickappend=tickappend,
        xrange=x_range,
        anno_text_formatter=text_formatter,
    )


def get_binary_cp_bar_plot(data, m):
    metric_name = m
    y_data = [y["cohort_short_name"] for y in data["cohorts"]]
    x_data = [int(x[metric_name] * 100) for x in data["cohorts"]]
    x_data = [[x, 100 - x] for x in x_data]
    legend = [m]
    tickvals = [0, 25, 50, 75, 100]
    ticktext = [str(x) + "%" for x in tickvals]

    return get_bar_plot(
        y_data,
        x_data,
        legend=legend,
        tickvals=tickvals,
        ticktext=ticktext,
        tickappend="%",
    )


def get_de_bar_plot(data):
    y_data = [
        y["label"] + "<br>" + str(int(y["population"] * 100)) + "% n"
        for y in data["classes"]
    ]
    x_data = [int(float(x["prediction_0_ratio"]) * 100) for x in data["classes"]]
    x_data = [[x, 100 - x] for x in x_data]
    legend = ['Predicted as "' + y["prediction_0_name"] + '"' for y in data["classes"]]
    tickvals = [0, 25, 50, 75, 100]
    ticktext = [str(x) + "%" for x in tickvals]

    return get_bar_plot(
        y_data,
        x_data,
        legend=legend,
        tickvals=tickvals,
        ticktext=ticktext,
        tickappend="%",
    )


def get_dot_plot(center, ep, em):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[center],
            y=[0.5],
            mode="markers",
            error_x=dict(
                type="data",
                symmetric=False,
                array=[ep],
                arrayminus=[em],
                color="rgba(39, 110, 237, 1)",
                width=15,
            ),
        )
    )
    fig.update_xaxes(
        showgrid=False,
        # tickvals=[0, 25, 50, 75, 100],
        # ticktext=["0%", "25%", "50%", "75%", "100%"],
        # range=[0, 100]
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=3,
        showticklabels=False,
    )
    fig.update_layout(height=240, width=700, plot_bgcolor="white")

    fig.update_traces(
        marker=dict(
            color="white", size=20, line=dict(width=2, color="rgba(39, 110, 237, 1)")
        )
    )

    png = pio.to_image(fig)
    png_base64 = base64.b64encode(png).decode("ascii")

    return png_base64


def get_bar_plot(
    y_data,
    x_data,
    legend=None,
    threshold=None,
    tickvals=None,
    ticktext=None,
    xrange=None,
    tickappend="",
    xtitle=None,
    anno_text_formatter=lambda x: str(round(x, 1)),
):
    fig = go.Figure()
    series = 0

    def get_colors(series, index):
        colors2 = ["rgba(39, 110, 237, 1)", "rgba(218, 227, 243, 1)"]
        return colors2[index % 2]

    def get_show_legend(legend, index):
        if not legend:
            return False
        if index == 0:
            return True
        return False

    for i in range(0, len(x_data[0])):
        for xd, yd in zip(x_data, y_data):
            fig.add_trace(
                go.Bar(
                    x=[xd[i]],
                    y=[yd],
                    width=0.9,
                    orientation="h",
                    marker=dict(
                        color=get_colors(series, i),
                        line=dict(color="rgb(248, 248, 249)", width=1),
                    ),
                    showlegend=get_show_legend(legend, series),
                )
            )
            series = series + 1

    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            zeroline=False,
            domain=[0.15, 1],
            tickvals=tickvals if tickvals else None,
            ticktext=ticktext if ticktext else None,
            range=xrange if xrange else None,
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        barmode="stack",
        bargap=0,
        paper_bgcolor="rgb(255, 255, 255)",
        plot_bgcolor="rgb(255, 2555, 255)",
        margin=dict(l=25, r=25, t=25, b=25),
    )

    annotations = []

    for yd, xd in zip(y_data, x_data):
        # labeling the y-axis
        annotations.append(
            dict(
                xref="paper",
                yref="y",
                x=0.14,
                y=yd,
                xanchor="right",
                text=str(yd),
                font=dict(family="Consolas", size=18, color="rgb(67, 67, 67)"),
                showarrow=False,
                align="right",
            )
        )
        # labeling the first percentage of each bar (x_axis)
        annotations.append(
            dict(
                xref="paper",
                yref="y",
                x=0.17,
                y=yd,
                text=anno_text_formatter(xd[0]) + tickappend,
                font=dict(family="Consolas", size=22, color="rgb(0, 0, 0)"),
                showarrow=False,
            )
        )

    # if xtitle:
    #     annotations.append(dict(
    #         xanchor="left",
    #         yanchor="bottom",
    #         x=0.2,
    #         y=-1,
    #                         text=xtitle,
    #                         font=dict(family='Consolas', size=22, color='rgb(0, 0, 0)'),
    #                         showarrow=False))

    fig.update_layout(
        annotations=annotations,
        width=700,
        height=len(x_data) * 70 + 30,
    )

    def get_legend_y(barchart_length):
        lookup = {
            1: 1.55,
            2: 1.27,
            3: 1.15,
            4: 1.11,
            5: 1.09,
            6: 1.07,
            7: 1.06,
            8: 1.05,
            9: 1.04,
            10: 1.03,
        }
        return lookup[barchart_length]

    fig.update_traces(cliponaxis=False)
    if threshold:
        fig.add_vline(
            x=threshold,
            annotation_text="Threshold",
            annotation_position="bottom right",
            line_width=3,
            line_dash="dash",
            line_color="red",
        )

    if legend:
        fig["data"][0]["name"] = legend[0]
        fig.update_layout(
            legend=dict(
                yanchor="top", y=get_legend_y(len(x_data)), xanchor="left", x=0.15
            )
        )

    png = pio.to_image(fig)
    png_base64 = base64.b64encode(png).decode("ascii")

    return png_base64


def get_de_box_plot(data):
    return get_box_plot(data)


def get_de_box_plot_image(data):
    processed_label = data
    for c in processed_label["data"]:
        c["label"] = (
            c["short_label"] + "<br>" + str(int(100 * round(c["population"], 3))) + "% n"
        )
        c["datapoints"] = c["prediction"]

    processed_label["data"] = list(reversed(processed_label["data"]))
    png_base64 = get_de_box_plot(processed_label)
    return div(
        img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div"
    )


def get_box_plot(data):
    fig = go.Figure()
    for i in data["data"]:
        fig.add_trace(
            go.Box(
                x=i["datapoints"],
                boxpoints=False,
                line_color="rgba(39, 110, 237, 1)",
                fillcolor="rgba(218, 227, 243, 1)",
                name=i["label"],
                showlegend=False,
            )
        )

    fig.update_layout(
        paper_bgcolor="rgb(255, 255, 255)",
        plot_bgcolor="rgb(255, 2555, 255)",
        margin=dict(l=25, r=25, t=25, b=25),
        width=700,
        height=len(data["data"]) * 70 + 60,
        # annotations=annotations,
        boxgap=0,
    )

    fig.update_layout()

    png = pio.to_image(fig)
    png_base64 = base64.b64encode(png).decode("ascii")

    return png_base64


def get_model_overview(data):
    model_left_items = []

    model_left_items.append(div(h3("Purpose"), p(data["ModelSummary"])))

    if data["ModelType"] == "binary_classification":
        model_left_items.append(
            div(
                p(
                    "Classification: {} vs {}".format(
                        data["classes"][0], data["classes"][1]
                    )
                )
            )
        )
    else:
        model_left_items.append(
            div(p("This is a {} model.".format(data["ModelType"].lower())))
        )

    model_left_items.append(
        div(
            h3("Model evaluation"),
            p(
                "This model is evaluated on a test set with {} datapoints.".format(
                    len(data["y_test"])
                )
            ),
        )
    )

    model_overview_left_container = div(model_left_items, _class="left_model_overview")

    model_main_items = []
    model_main_items.extend(
        [
            h3("Target values"),
            p(
                "Here are your defined target values for your model "
                "performance and/or other model assessment parameters:"
            ),
        ]
    )

    metric_targets_elems = []
    for item in data["metrics_targets"]:
        metric_targets_elems.append(li(item))

    model_main_items.append(
        div(
            ul(metric_targets_elems),
            _style="border: 2px solid black; border-radius: 5px;",
        )
    )

    model_overview_main_container = div(model_main_items, _class="main_model_overview")

    heading = [h1(data["ModelName"])]
    if data["runinfo"]:
        heading.append(
            p(
                "Generated by {} on {}".format(
                    data["runinfo"]["submittedBy"], data["runinfo"]["startTimeUtc"]
                )
            )
        )
        heading.append(
            p(
                "Source RAI dashboard: ",
                a(
                    data["runinfo"]["dashboard_title"],
                    _href=data["runinfo"]["dashboard_link"],
                ),
            )
        )
        heading.append(p(f"Model id: {data['runinfo']['model_id']}"))

    model_overview_container = div(
        div(heading, _class="header"),
        get_page_divider("Model Summary"),
        model_overview_left_container,
        model_overview_main_container,
        _class="container",
    )

    return model_overview_container


def get_cohorts_page(data, metric_config):
    left_elems = div(
        p("Observe evidence of model performance across your passed cohorts:"),
        _class="left",
    )
    cp_heading_main = div(_class="main")
    heading_section = str(
        div(
            get_page_divider("Cohorts"), left_elems, cp_heading_main, _class="container"
        )
    )

    # start section for each predefined cohort
    def populate_cp_container(key, container, m, data):
        if key not in data.keys():
            return

        def get_regression_bar_plot(d, m):
            first_data_point = next(iter(d), None)
            threshold = None
            if first_data_point:
                threshold = first_data_point.get("threshold", None)
            y_data = [
                [y["short_label"], str(int(y["population"] * 100)) + "% n"] for y in d
            ]
            y_data = ["<br>".join(y) for y in y_data]
            x_data = [x[m] for x in d]
            if m in ["accuracy_score", "recall_score", "precision_score", "f1_score"]:
                max_x = 1
            else:
                max_x = max(x_data)
            x_data = [[x, max_x - x] for x in x_data]
            legend = [m]

            return get_bar_plot(
                list(reversed(y_data)),
                list(reversed(x_data)),
                legend=legend,
                threshold=threshold,
            )

        message_lookup = {
            "cohorts": {
                "left_heading": "My Cohorts",
                "main_heading": "My prebuilt dataset cohorts",
            },
            "error_analysis_max": {
                "left_heading": "Highest ranked cohorts",
                "main_heading": "Highest ranked cohorts",
            },
            "error_analysis_min": {
                "left_heading": "Lowest ranked cohorts",
                "main_heading": "Lowest ranked cohorts",
            },
        }

        filtered_data = [d for d in data[key] if m in d.keys()]
        if len(filtered_data) == 0:
            return

        container[key].append(
            get_cohorts_performance_container(
                filtered_data, m, get_regression_bar_plot, message_lookup[key]
            )
        )

    cohort_performance_containers = {
        "cohorts": [],
        "error_analysis_max": [],
        "error_analysis_min": [],
    }
    for k in ["cohorts", "error_analysis_max", "error_analysis_min"]:
        for m in metric_config:
            populate_cp_container(k, cohort_performance_containers, m, data)

    cohort_performance_section = "".join(cohort_performance_containers["cohorts"])
    cohort_performance_section = cohort_performance_section + "".join(
        cohort_performance_containers["error_analysis_max"]
    )
    cohort_performance_section = cohort_performance_section + "".join(
        cohort_performance_containers["error_analysis_min"]
    )

    return str(heading_section + cohort_performance_section)


def get_causal_page(data):
    left_elem = [
        div(
            p(
                "ausal analysis answers real-world what-if questions "
                "about how changing specific treatments would impact outcomes."
            )
        )
    ]

    left_container = div(left_elem, _class="left")

    main_elems = []

    def get_causal_dot_plot(center, em, ep):
        png_base64 = get_dot_plot(center, em, ep)
        return div(
            img(_src="data:image/png;base64,{}".format(png_base64)), _class="image_div"
        )

    def get_table_row(data):
        table_row_elems = []
        for v in data:
            table_row_elems.append(td(v, _class="cell"))
        return tr(table_row_elems, _class="row")

    def get_table(data):
        horizontal_headings = [
            "Index",
            "Current<br>Value",
            "Recommended<br>Treatment",
            "Effect<br>Estimate",
        ]
        headings_td = [td(x, _class="header_cell") for x in horizontal_headings]
        headings = thead(tr(headings_td, _class="row"), _class="table-head")

        rows_elems = []
        for elem in data:
            rows_elems.append(get_table_row(elem))

        body = tbody(rows_elems, _class="table-body")

        return table(headings, body, _class="table")

    for f in data["global_effect"].values():
        main_elems.append(
            div(
                h3(f["feature"]),
                p(
                    'On average, increasing "{}" by 1 unit increases the outcome by {}'.format(
                        f["feature"], round(f["point"], 3)
                    )
                ),
                get_causal_dot_plot(
                    f["point"], f["ci_upper"] - f["point"], f["point"] - f["ci_lower"]
                ),
                _class="nobreak_div",
            )
        )

        main_elems.append(
            h3(
                'Top data points responding the most to treatment on "{}":'.format(
                    f["feature"]
                )
            )
        )
        main_elems.append(
            p(
                "Datapoints with the largest estimated causal responses to treatment feature: "
                '"{}"'.format(f["feature"])
            )
        )

        def causal_policies_map_to_table(policy):
            ct = policy["Current treatment"]
            et = policy["Effect of treatment"]

            ct = round(ct, 2) if isinstance(ct, (int, float)) else ct
            et = round(et, 2) if isinstance(et, (int, float)) else et

            return [
                policy["index"],
                ct,
                policy["Treatment"],
                et,
            ]

        main_elems.append(
            get_table(
                list(
                    map(
                        causal_policies_map_to_table,
                        data["top_local_policies"][f["feature"]],
                    )
                )
            )
        )

    main_container = div(main_elems, _class="main")

    return div(
        div(
            get_page_divider("Causal Inference"),
            left_container,
            main_container,
            _class="nobreak_div",
        ),
        _class="container nobreak_div",
    )


def to_pdf(html, output, wkhtmltopdf_path=None):
    options = {
        "page-size": "Letter",
        "margin-top": "0.75in",
        "margin-right": "0.75in",
        "margin-bottom": "0.75in",
        "margin-left": "0.75in",
        "encoding": "UTF-8",
    }

    wkhtmlconfig = None
    if wkhtmltopdf_path:
        wkhtmlconfig = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)

    pdfkit.from_string(
        html, output_path=output, options=options, configuration=wkhtmlconfig
    )
