a = {
    "metrics":
    {
        "data_drift": {
            "value": 1
        },
        "data_quality":{
            "value":2,
            "threshold":0.2
        }
    }
}

b = {
    "metrics":
    {
        "data_drift": {
            "value": 923,
            "samples": {
                "uri": "bla"
            }
        }
    }
}

def merge(source_dict, new_dict):
    result = source_dict
    
    if type(source_dict) is not dict or type(new_dict) is not dict:
        print(source_dict, new_dict)
        raise Exception(f"Error: Attempting to merge non-dictionary objects {source_dict} and {new_dict}.")

    for key in new_dict:
        if key in source_dict:
            result[key] = _merge(source_dict[key], new_dict[key], key)
        else:
            result[key] = new_dict[key]

    return result

def _merge(source_dict, new_dict, context):
    result = source_dict
    
    if type(source_dict) is not dict or type(new_dict) is not dict:
        print(source_dict, new_dict)
        raise Exception(f"Error: A duplicate key is found for entries {{{context}: {new_dict}}} and"
                                        + f" {{{context}: {source_dict}}}")

    for key in new_dict:
        if key in source_dict:
            result[key] = _merge(source_dict[key], new_dict[key], f"{context}.{key}")
        else:
            result[key] = new_dict[key]

    return result

c = merge(a,b)

print(c)