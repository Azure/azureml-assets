# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from domonic.html import div, h3, p, img, table, td, th, tr, ul, li, thead, tbody
from . import common_components as cc
from ._rai_insight_data import get_metric


def get_model_overview_page(data):
    return cc.get_model_overview(data)


def get_data_explorer_page(data):
    de_heading_left_elms = p(
        "Evaluate your dataset to assess representation of identified cohorts:"
    )

    de_heading_left_container = div(de_heading_left_elms, _class="left")

    de_containers = [div(de_heading_left_container, _class="container")]

    def get_de_bar_plot(data):
        class_0 = data[0]["prediction"][0]

        y_data = [
            str(c["short_label"]) + "<br>" + str(int(100 * c["population"])) + "% n"
            for c in data
        ]
        x_data = [
            100 * (list(c["prediction"]).count(class_0)) / (len(c["prediction"]))
            for c in data
        ]
        x_data = [[x, 100 - x] for x in x_data]

        tickvals = [0, 25, 50, 75, 100]
        ticktext = [str(x) + "%" for x in tickvals]
        legend = ["Predicted as {}".format(class_0)]

        png_base64 = cc.get_bar_plot(
            list(reversed(y_data)),
            list(reversed(x_data)),
            legend=legend,
            tickvals=tickvals,
            ticktext=ticktext,
            tickappend="%",
        )
        return div(
            img(_src="data:image/png;base64,{}".format(png_base64)),
            _class="image_div",
        )

    def get_de_containers(c):
        feature_list = ul()
        for d in c["data"]:
            feature_list.append(li("{}: {}".format(d["short_label"], d["label"])))

        left_container = div(h3(c["feature_name"]), feature_list, _class="left")
        de_main_elems = []

        de_main_elems.append(h3(c["feature_name"]))

        for i in c["data"]:
            de_main_elems.append(
                p(
                    'Cohort "{}" has {}% {}'.format(
                        i["label"],
                        round(i[c["primary_metric"]] * 100, 1),
                        c["primary_metric"],
                    )
                )
            )
        de_main_elems.append(
            div(
                p(
                    "Predicted classification output of the different cohorts are as follows:"
                ),
                get_de_bar_plot(c["data"]),
                _class="nobreak_div",
            )
        )

        main_container = div(de_main_elems, _class="main")

        return div(left_container, main_container, _class="container")

    for c in data:
        de_containers.append(get_de_containers(c))

    return str(div(cc.get_page_divider("Data Explorer"), de_containers))


def _get_model_performance_explanation_text(metric, data):
    score_value = str(int(100 * data["metrics"][metric]))
    pos_label = data["pos_label"]
    neg_label = data["neg_label"]
    y_test_size = len(data["y_test"])
    if metric == "accuracy_score":
        tp = data["confusion_matrix"]["tp"]
        tn = data["confusion_matrix"]["tn"]
        total = len(data["y_pred"])
        return div(
            h3("Accuracy"),
            p(
                "{}% of data points have the correct prediction.<br>".format(
                    score_value
                )
            ),
            p(
                "Accuracy = correct predictions / all predictions<br>"
                "= ({} + {}) / {}".format(tp, tn, total)
            ),
        )
    elif metric == "recall_score":
        return div(
            h3("{}% Recall".format(score_value)),
            p(
                '{}% of data points that are actually "{}" are likely to be predicted as "{}"'.format(
                    score_value, pos_label, pos_label
                )
            ),
        )
    elif metric == "precision_score":
        return div(
            h3("{}% Precision".format(score_value)),
            p(
                '{}% of data points predicted as "{}", are likely to actually be "{}"'.format(
                    score_value, pos_label, neg_label
                )
            ),
        )
    elif metric == "false_negative":
        adjusted_score = int(round(100 * data["metrics"][metric] / y_test_size))
        return div(
            h3("{}% False Negative".format(adjusted_score)),
            p(
                '{}% of data points that are predicted as "{}" should have been predicted as "{}"'.format(
                    adjusted_score, neg_label, pos_label
                )
            ),
        )
    elif metric == "false_positive":
        adjusted_score = int(round(100 * data["metrics"][metric] / y_test_size))
        return div(
            h3("{}% False Positive".format(adjusted_score)),
            p(
                '{}% of data points that are predicted as "{}" should have been predicted as "{}"'.format(
                    adjusted_score, pos_label, neg_label
                )
            ),
        )
    else:
        return div()


def _get_confusion_matrix_grid(data):
    cm = data["confusion_matrix"]
    negative = data["classes"][0]
    positive = data["classes"][1]
    return table(
        tr(
            th(_class="header_cell"),
            th('Actual<br>"{}"'.format(positive), _class="header_cell"),
            th('Actual<br>"{}"'.format(negative), _class="header_cell"),
        ),
        tr(
            th('Predicted<br>"{}"'.format(positive), _class="header_cell"),
            td(
                p(
                    cm["tp"],
                    _style="font-size:22px; color:#107C10; text-align: center;",
                ),
                p(
                    "correct prediction",
                    _style="font-size:14px; text-align: center;",
                ),
                _class="cell",
            ),
            td(
                cm["fn"],
                _class="cell",
                _style="font-size:22px; color:#A80000; text-align: center;",
            ),
        ),
        tr(
            th('Predicted<br>"{}"'.format(negative), _class="header_cell"),
            td(
                cm["fp"],
                _class="cell",
                _style="font-size:22px; color:#A80000; text-align: center;",
            ),
            td(
                p(
                    cm["tn"],
                    _style="font-size:22px; color:#107C10; text-align: center;",
                ),
                p(
                    "correct prediction",
                    _style="font-size:14px; text-align: center;",
                ),
                _class="cell",
            ),
        ),
        _style="width: 5in",
    )


def get_model_performance_page(data):
    left_metric_elems = [p("Observe evidence of your model performance here:")]

    def get_metric_bar_plot(mname, data):
        y_0_filtermap = [
            True if i == data["classes"][0] else False for i in data["y_test"]
        ]
        y_1_filtermap = [
            True if i == data["classes"][1] else False for i in data["y_test"]
        ]
        class_0_metric = get_metric(
            mname,
            data["y_test"][y_0_filtermap],
            data["y_pred"][y_0_filtermap],
            pos_label=data["classes"][0],
            labels=data["classes"],
        )
        class_1_metric = get_metric(
            mname,
            data["y_test"][y_1_filtermap],
            data["y_pred"][y_1_filtermap],
            pos_label=data["classes"][1],
            labels=data["classes"],
        )

        y_data = [
            'Acutal "{}"'.format(data["classes"][0]),
            'Acutal "{}"'.format(data["classes"][1]),
        ]
        x_data = [int(class_0_metric * 100), int(class_1_metric * 100)]
        x_data = [[x, 100 - x] for x in x_data]

        legend = [m]
        tickvals = [0, 25, 50, 75, 100]
        ticktext = [str(x) + "%" for x in tickvals]

        png_base64 = cc.get_bar_plot(
            y_data,
            x_data,
            legend=legend,
            tickvals=tickvals,
            ticktext=ticktext,
            tickappend="%",
        )
        return div(
            img(_src="data:image/png;base64,{}".format(png_base64)),
            _class="image_div",
        )

    main_elems = []
    main_elems.append(_get_confusion_matrix_grid(data))

    for m in data["metrics"]:
        left_metric_elems.append(_get_model_performance_explanation_text(m, data))

    left_container = div(left_metric_elems, _class="left")
    main_container = div(main_elems, _class="main")
    return str(
        div(
            cc.get_page_divider("Model Performance"),
            left_container,
            main_container,
            _class="container",
        )
    )


def get_cohorts_page(data, metrics_config):
    return cc.get_cohorts_page(data, metrics_config)


def get_feature_importance_page(data):
    return cc.get_feature_importance_page(data)


def get_causal_page(data):
    return cc.get_causal_page(data)


def get_fairlearn_page(data):
    heading = div(
        p(
            "Understand your model's fairness issues "
            "using group-fairness metrics across sensitive features and cohorts. "
            "Pay particular attention to the cohorts who receive worse treatments "
            "(predictions) by your model."
        ),
        _class="left",
    )

    left_elems = []

    for f in data:
        left_elems.append(h3('Feature "{}"'.format(f)))
        metric_section = []
        for metric_key, metric_details in data[f]["metrics"].items():
            if metric_key in ["false_positive", "false_negative"]:
                continue
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

    def get_fairness_bar_plot(data):
        y_data = [
            str(c) + "<br>" + str(int(100 * data[c]["population"])) + "% n"
            for c in data
        ]
        x_data = [
            100
            * (
                get_metric("selection_rate", data[c]["y_test"], data[c]["y_pred"]),
                data[c]["pos_label"],
            )
            for c in data
        ]
        x_data = [[x, 100 - x] for x in x_data]

        tickvals = [0, 25, 50, 75, 100]
        ticktext = [str(x) + "%" for x in tickvals]

        png_base64 = cc.get_bar_plot(
            y_data, x_data, tickvals=tickvals, ticktext=ticktext, tickappend="%"
        )

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
        horizontal_headings = [d.replace("_", "<br>") for d in metric_list]
        vertical_headings = list(data["statistics"].keys())

        headings_td = [td(_class="header_cell")] + [
            td(x, _class="header_cell") for x in horizontal_headings
        ]
        headings = thead(tr(headings_td, _class="row"), _class="table-head")

        rows_elems = []
        for vh in vertical_headings:
            row_data = []
            for m in metric_list:
                row_data.append(round(data["metrics"][m]["group_metric"][vh], 2))
            rows_elems.append(get_table_row(vh, row_data))

        body = tbody(rows_elems, _class="table-body")

        return table(headings, body, _class="table")

    main_elems = []
    # Selection rate
    for f in data:
        main_elems.append(
            div(
                h3("Selection rate"),
                get_fairness_bar_plot(data[f]["statistics"]),
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
            cc.get_page_divider("Fairness Assessment"),
            div(heading, _class="container"),
            div(left_container, main_container, _class="container"),
            _class="container nobreak_div",
        )
    )
