import uuid
from pyspark.sql.functions import col, udf, sum, avg, percentile_approx, min, count, lit
from pyspark.sql.types import StringType

def impute_ids_for_failed_calls(token_df):
    """Impute ids for failed calls
    if the id string is null or empty, then generate a new id using the following logic:
    - if the status_code is not 200, then generate a new guid string as id
    """
    # if the id is null, then generate a new id using the following logic:
    # - if the status_code is not 200, then generate a new guid as id

    token_df = token_df.withColumn(
        "id",
        udf(
            lambda id, status_code: id if ((id is not None) or (status_code == 200))
              else uuid.uuid4().hex,
            StringType(),
        )(col("id"), col("status_code"))
    )

    return token_df

def check_data_quality(token_df):
    """check for any violation of assumptions about the data
    - check that the token_df columns contains: 
        ['node_id': id the of the node in prompt flow that is calling the LLM. 
         'id': request id for the call.
         'status_code': the status code returned by the API call. Need for call success and RAI metrics.
         'completion_tokens': token count for the final response. 
         'prompt_tokens': token count for the prompt in the request.
         ]
    - check that status_code, id column has no null values
    - check that for rows where status_code is 200, no column has a null value
    """
    # check that the token_df columns contains these columns:
    expected_columns = [
        "node_id",
        "id",
        "status_code",
        "completion_tokens",
        "prompt_tokens",
    ]

    if set(expected_columns).issubset(token_df.columns) == False:
        print(f"token_df columns are not as expected. Expected: {expected_columns}, ")
        print(f"Actual: {token_df.columns}")
        return 

    # check that status_code, id and node_id column has no null values
    null_status_code_count =  token_df.filter(token_df["status_code"].isNull()).count()
    if null_status_code_count != 0:
        print(f"token_df contains {null_status_code_count} null values in status_code column")
        print(f"filtering out the rows with null status_code")
        # filter out the rows with null status_code
        token_df = token_df.filter(token_df["status_code"].isNotNull())
        
    null_id_count =  token_df.filter(token_df["id"].isNull()).count()
    if null_id_count != 0:
        print(f"token_df contains {null_id_count} null values in id column")
        print(f"filtering out the rows with null id")
        # filter out the rows with null id
        token_df = token_df.filter(token_df["id"].isNotNull())
    
    null_node_id_count =  token_df.filter(token_df["node_id"].isNull()).count()
    if null_node_id_count != 0:
        print(f"token_df contains {null_node_id_count} null values in node_id column")
        print(f"filtering out the rows with null node_id")
        # filter out the rows with null node_id
        token_df = token_df.filter(token_df["node_id"].isNotNull())

    # check that for rows where status_code is 200, no expected column has a null value. If so, filter out those rows
    for column in expected_columns:
        if column == "status_code":
            continue
        null_count = token_df.filter(token_df["status_code"] == 200).filter(
            token_df[column].isNull()
        ).count()
        if null_count != 0:
            print(f"token_df contains {null_count} null values in {column} column for rows where status_code is 200")
            print(f"filtering out the rows with null {column}")
            # filter out the rows with null column
            token_df = token_df.filter(token_df[column].isNotNull())
    return token_df

def compute_conditional_counts_df(token_df, dimensions, condition_column, condition_value, metric_name):
    '''
    Needs the token_df with columns dimensions, condition_column
    Returns a metric df with columns group, group_pivot, metric_name, metric_value
    '''
    # check if the token_df has the expected columns
    if set(dimensions + [condition_column]).issubset(token_df.columns) == False:
        return
    
    return token_df.filter(token_df[condition_column] == condition_value).groupBy(dimensions).agg(
    lit(metric_name).alias('metric_name'),
    count('*').cast("float").alias("metric_value"),
    )

def compute_avg_df(token_df, columns, dimensions, metric_prefix=""):
    '''
    Needs the token_df with columns dimensions, columns
    Returns a metric df with columns group, group_pivot, metric_name, metric_value
    where metric_name is the column name and metric_value is the average of the column
    '''
    # check if the token_df has the expected columns
    if set(dimensions + columns + ['status_code']).issubset(token_df.columns) == False:
        return
    # restrict the token_df to only expected columns
    token_df = token_df.select(dimensions + columns + ['status_code'])
    
    avg_columns = [avg(col).cast("float").alias(f"{metric_prefix}_avg_{col}") for col in columns]
    stack_columns = ["'"+ metric_prefix +"_avg_" + col + "', " + metric_prefix + "_avg_" + col for col in columns]
    stack_expr = "stack(" + str(len(columns)) + ", " + ", ".join(stack_columns) + ") as (metric_name, metric_value)"
    return token_df.filter(token_df["status_code"] == 200).groupBy(dimensions).agg(*avg_columns).selectExpr("group", "group_pivot", stack_expr)

def compute_sum_df(token_df, columns, dimensions, metric_prefix=""):
    '''
    Needs the token_df with columns dimensions, columns
    Returns a metric df with columns group, group_pivot, metric_name, metric_value
    where metric_name is the column name and metric_value is the sum of the column
    '''
    # check if the token_df has the expected columns
    if set(dimensions + columns + ['status_code']).issubset(token_df.columns) == False:
        return
    # restrict the token_df to only expected columns
    token_df = token_df.select(dimensions + columns + ['status_code'])

    sum_columns = [sum(col).cast("float").alias(f"{metric_prefix}_sum_{col}") for col in columns]
    stack_columns = ["'"+ metric_prefix +"_sum_" + col + "', " + metric_prefix + "_sum_" + col for col in columns]
    stack_expr = "stack(" + str(len(columns)) + ", " + ", ".join(stack_columns) + ") as (metric_name, metric_value)"
    return token_df.filter(token_df["status_code"] == 200).groupBy(dimensions).agg(*sum_columns).selectExpr("group", "group_pivot", stack_expr)


def compute_percentage_metric_df(numerator_metric_df, denominator_metric_df, ratio_metric_name, dimensions):
    '''
    Needs the metric df with columns group, group_pivot, metric_name, metric_value. 
    Returns a metric df with columns group, group_pivot, metric_name, metric_value,
    where metric_name is the ratio_metric_name and 
    metric_value is the ratio of the metric_value of the numerator_metric_df to the metric_value of the denominator_metric_df
    '''
    ratio_df = numerator_metric_df.withColumnRenamed("metric_value","numerator")\
    .join(
            denominator_metric_df.withColumnRenamed("metric_value","denominator"), on=dimensions, how="inner",
        ).select(dimensions+["numerator","denominator"])
    ratio_df = ratio_df.withColumn(
            "metric_value",
            ratio_df["numerator"]
            / ratio_df["denominator"]
            * 100,
        )
    ratio_df = ratio_df.withColumn(
            "metric_name", lit(ratio_metric_name)
        )
    ratio_df = ratio_df.select(
            dimensions + ["metric_name", "metric_value"]
        )
    return ratio_df

def compute_GPU_utilization_metrics(token_df):
    """Compute GPU utilization metrics for each group and group_pivot:
    - Expect the token_df to have the following columns:group, group_pivot
    - Calls failed due to overload: 
        -- total number of calls with 429 status code
        -- %age of calls with 429 status code
    - Calls Succeeded:
        -- total number of calls with 200 status code
        -- %age of calls with 200 status code
    - GPU utlization for the successful calls
        -- Sum of prompt_tokens, completion_tokens, total_tokens for calls with 200 status code
        -- Average of prompt_tokens, completion_tokens, total_tokens for calls with 200 status code
    """
    # check if the token_df has the expected dimension columns
    dimensions=['group','group_pivot']
    if set(dimensions).issubset(token_df.columns) == False:
        return

    # Call Counts
    call_counts = token_df.groupBy(dimensions).agg(
    lit("num_calls").alias('metric_name'),
    count('*').cast("float").alias("metric_value"),
    )
    
    # Calls failed due to overload
    
    calls_failed_due_to_overload =compute_conditional_counts_df(
        token_df=token_df,
        dimensions=dimensions,
        condition_column="status_code",
        condition_value=429,
        metric_name="num_calls_with_status_code_429"
    )

    # Percentage calls failed due to overload
    percent_calls_failed_due_to_overload = compute_percentage_metric_df(
        numerator_metric_df=calls_failed_due_to_overload,
        denominator_metric_df=call_counts,
        ratio_metric_name="percentage_calls_with_status_code_429",
        dimensions=dimensions
    )

    # Calls Succeeded
    calls_succeeded = compute_conditional_counts_df(
        token_df=token_df,
        dimensions=dimensions,
        condition_column="status_code",
        condition_value=200,
        metric_name="num_calls_with_status_code_200"
    )

    # Percentage calls succeeded
    percentage_calls_succeeded = compute_percentage_metric_df(
        numerator_metric_df=calls_succeeded,
        denominator_metric_df=call_counts,
        ratio_metric_name="percentage_calls_with_status_code_200",
        dimensions=dimensions
    )

    # GPU utlization for the successful calls
    gpu_utilization_sum = compute_sum_df(
        token_df=token_df,
        columns=["prompt_tokens", "completion_tokens", "total_tokens"],
        dimensions=dimensions,
        metric_prefix="gpu_utilization"
    )
    

    gpu_utilization_avg = compute_avg_df(
        token_df=token_df,
        columns=["prompt_tokens", "completion_tokens", "total_tokens"],
        dimensions=dimensions,
        metric_prefix="gpu_utilization"
    )
    
    # Union all the metric dfs
    gpu_utilization_metrics = call_counts.unionAll(
        calls_failed_due_to_overload
    ).unionAll(
        percent_calls_failed_due_to_overload
    ).unionAll(
        calls_succeeded
    ).unionAll(
        percentage_calls_succeeded
    ).unionAll(
        gpu_utilization_sum
    ).unionAll(
        gpu_utilization_avg
    )
    return gpu_utilization_metrics

def compute_GPU_waste_metrics(token_df):
    """Compute GPU waste metrics:
    - Calls wasted due to truncation
        -- total number of calls with finish_reason as "length"
        -- %age of calls with finish_reason as "length"
        -- Sum of prompt_tokens, completion_tokens, total_tokens for calls with finish_reason as "length"
        -- Average of prompt_tokens, completion_tokens, total_tokens for calls with finish_reason as "length"
        -- %age of total tokens wasted due to truncation
    - Waste due to unused reserved tokens
        -- min, 1st percentile and 5th percentile of max_tokens-completion_tokens for calls with 200 status code
        -- %age of reserved tokens wasted
    """
    # check if the token_df has the expected dimension columns
    dimensions=['group','group_pivot']
    if set(dimensions).issubset(token_df.columns) == False:
        return
    
    # These metrics are computed if we have max_tokens and finish_reason in the dataset
    if ("finish_reason" not in token_df.columns) or ("max_tokens" not in token_df.columns):
        return

    # Calls wasted due to truncation
    calls_succeeded = compute_conditional_counts_df(
        token_df=token_df,
        dimensions=dimensions,
        condition_column="status_code",
        condition_value=200,
        metric_name="num_calls_with_status_code_200"
    )

    calls_wasted_due_to_truncation =compute_conditional_counts_df(
        token_df=token_df,
        dimensions=dimensions,
        condition_column="finish_reason",
        condition_value="length",
        metric_name="num_calls_with_finish_reason_length"
    )

    
    percentage_calls_wasted_due_to_truncation = compute_percentage_metric_df(
        numerator_metric_df=calls_wasted_due_to_truncation,
        denominator_metric_df=calls_succeeded,
        ratio_metric_name="percentage_calls_wasted_due_to_truncation",
        dimensions=dimensions
    )
    
    # Waste due to truncation of response. This happens when the max_tokens is set to a value so small that the response is truncated. 
    gpu_waste_sum_truncation = compute_sum_df(
        token_df=token_df,
        columns=["prompt_tokens", "completion_tokens", "total_tokens"],
        dimensions=dimensions,
        metric_prefix="gpu_waste_due_to_response_truncation"
    )
    
    gpu_waste_avg_truncation = compute_avg_df(
        token_df=token_df,
        columns=["prompt_tokens", "completion_tokens", "total_tokens"],
        dimensions=dimensions,
        metric_prefix="gpu_waste_due_to_response_truncation"
    )
    
    # GPU utlization for the successful calls
    gpu_utilization_sum = compute_sum_df(
        token_df=token_df,
        columns=["prompt_tokens", "completion_tokens", "total_tokens"],
        dimensions=dimensions
    )

    percentage_total_tokens_wasted_due_to_truncation = compute_percentage_metric_df(
        numerator_metric_df=gpu_waste_sum_truncation,
        denominator_metric_df=gpu_utilization_sum,
        ratio_metric_name="percentage_total_tokens_wasted_due_to_truncation",
        dimensions=dimensions
    )

    # Waste due to unused reserved tokens. 
    # TODO: There are new developments in AOAI that mitigate some of this waste. Need to update this metric accordingly. 
    tail_waste_due_to_unused_reserved_tokens = token_df.filter(token_df["status_code"] == 200).groupBy(dimensions)\
        .agg(
        min("unused_reserved_tokens").cast("float").alias("min_unused_reserved_tokens"), 
        percentile_approx("unused_reserved_tokens", 0.01).cast("float").alias("1st_percentile_unused_reserved_tokens"),
        percentile_approx("unused_reserved_tokens", 0.05).cast("float").alias("5th_percentile_unused_reserved_tokens")
    ).selectExpr("group", "group_pivot", "stack(3, 'min_unused_reserved_tokens', min_unused_reserved_tokens,\
                  '1st_percentile_unused_reserved_tokens', 1st_percentile_unused_reserved_tokens,\
                  '5th_percentile_unused_reserved_tokens', 5th_percentile_unused_reserved_tokens) as (metric_name, metric_value)")

    sum_waste_due_to_unused_reserved_tokens = compute_sum_df(
        token_df=token_df,
        columns=["unused_reserved_tokens"],
        dimensions=dimensions,
        metric_prefix="gpu_waste"
    )

    sum_reserved_tokens = compute_sum_df(
        token_df=token_df,
        columns=["max_tokens"],
        dimensions=dimensions,
        metric_prefix="reserved_tokens"
    )

    percentage_reserved_tokens_wasted = compute_percentage_metric_df(
        numerator_metric_df=sum_waste_due_to_unused_reserved_tokens,
        denominator_metric_df=sum_reserved_tokens,
        ratio_metric_name="percentage_reserved_tokens_wasted",
        dimensions=dimensions
    )

    # Union all the metric dfs
    gpu_waste_metrics = calls_wasted_due_to_truncation.unionAll(
        percentage_calls_wasted_due_to_truncation
    ).unionAll(
        gpu_waste_sum_truncation
    ).unionAll(
        gpu_waste_avg_truncation
    ).unionAll(
        percentage_total_tokens_wasted_due_to_truncation
    ).unionAll(
        tail_waste_due_to_unused_reserved_tokens
    ).unionAll(
        sum_waste_due_to_unused_reserved_tokens
    ).unionAll(
        sum_reserved_tokens
    ).unionAll(
        percentage_reserved_tokens_wasted
    )
    return gpu_waste_metrics