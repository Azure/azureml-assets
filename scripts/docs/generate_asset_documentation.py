# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
import snakemd
import re
import os


def create_asset_doc(spec_file_name, asset_type):
    # read the asset yaml file and parse it
    yaml = YAML()
    with open(spec_file_name, 'r') as f:
        asset = yaml.load(f)

    asset_type = str(asset_type).replace("AssetType.", "").lower()

    # create asset document and add info
    doc = snakemd.new_doc()
    doc = add_intro(doc, asset)
    doc = add_inputs(doc, asset)
    doc = add_outputs(doc, asset)
    doc = add_additional_details(doc, asset)

    # check if asset folder exists, create one if not
    if not os.path.exists(f"{asset_type}s/"):
        os.mkdir(asset_type + "s")

    asset_file_name = f"{asset_type}-{asset['name']}"

    # write the file into the asset folder
    with open(f"{asset_type}s/{asset_file_name}.md", 'w') as f:
        f.write(str(doc))

    return asset_type, asset["name"], asset_file_name


def add_intro(doc, asset):

    doc.add_heading(asset['display_name'] if "display_name" in asset else asset["name"], level=2)

    doc.add_heading("README file ", level=3)

    doc.add_heading("Overview ", level=3)

    if "description" in asset:
        doc.add_paragraph("**Description**: " + str(asset['description']))
    if "version" in asset:
        doc.add_paragraph("**Version**: " + str(asset['version']))
    if "type" in asset:
        doc.add_paragraph("**Type**: " + asset['type'])
    if "tags" in asset and "license" in asset["tags"]:
        doc.add_paragraph("**License**: " + asset["tags"]["license"])

    doc.add_heading("YAML Syntax ", level=3)

    if "type" in asset and asset["type"] == "automl":
        doc.add_paragraph("**Task**: " + asset["task"] if "task" in asset else "")

    if "properties" in asset:
        doc.add_heading("Properties", level=4)
        if "SHA" in asset["properties"]:
            doc.add_paragraph("**SHA**: " + asset["properties"]["SHA"])
        if "datasets" in asset["properties"]:
            doc.add_paragraph("**Datasets**: " + asset["properties"]["datasets"])
        if "finetuning-tasks" in asset["properties"]:
            doc.add_paragraph("**Finetuning Tasks**: " + asset["properties"]["finetuning-tasks"])
        if "languages" in asset["properties"]:
            doc.add_paragraph("**Languages**: " + asset["properties"]["languages"])

    return doc


def add_additional_details(doc, asset):
    doc.add_heading("Parameters ", level=3)

    doc.add_heading("Code", level=3)
    if "type" in asset and asset["type"] == "command" and "code" in asset:
        doc.add_paragraph(asset['code'])

    doc.add_heading("Environment ", level=3)
    doc.add_paragraph(asset['environment'] if "environment" in asset else "")

    doc.add_heading("Compute Specifications ", level=3)

    if "tags" in asset and "min_inference_sku":
        doc.add_paragraph(asset['tags']['min_inference_sku'] in asset["tags"])

    return doc


# set attributes
def get_comments_map(self, key):
    coms = []
    comments = self.ca.items.get(key)
    if comments is None:
        return coms
    for token in comments:
        if token is None:
            continue
        elif isinstance(token, list):
            coms.extend(token)
        else:
            coms.append(token)
    return coms


def get_comments_seq(self, idx):
    coms = []
    comments = self.ca.items.get(idx)
    if comments is None:
        return coms
    for token in comments:
        if token is None:
            continue
        elif isinstance(token, list):
            coms.extend(token)
        else:
            coms.append(token)
    return coms


setattr(CommentedMap, 'get_comments', get_comments_map)
setattr(CommentedSeq, 'get_comments', get_comments_seq)


def check_comments(data):
    if isinstance(data, CommentedMap):
        for k, v in data.items():
            comments = data.get_comments(k)
    elif isinstance(data, CommentedSeq):
        for idx, item in enumerate(data):
            comments = data.get_comments(k)
    for comment in comments:
        text = comment.value
        text = re.sub(r"[\n\t\s]*", "", text)
        if text:
            return True
    return False


def insert_comments_between_inputs(doc, data):
    if isinstance(data, CommentedMap):
        for k, v in data.items():
            comments = data.get_comments(k)
    elif isinstance(data, CommentedSeq):
        for idx, item in enumerate(data):
            comments = data.get_comments(k)
    for comment in comments:
        text = comment.value
        text = re.sub(r"#", "", text)
        doc.add_paragraph(text)

    return doc


def insert_comments_under_input(doc, data):
    for comments in data:
        if comments:
            for comment in comments:
                doc.add_paragraph(comment.value.strip('#'))
    return doc


def add_inputs(doc, asset):
    doc.add_heading("Inputs ", level=2)

    if "inputs" in asset:
        headers = ['Name', 'Description', 'Type', 'Default', 'Optional', 'Enum']
        rows = []
        if asset.ca.items.get('inputs') is not None:
            doc = insert_comments_under_input(doc, asset.ca.items.get('inputs'))

        for k, v in asset['inputs'].items():
            row = []
            row.append(k)
            row.append(v['description'] if 'description' in v else ' ')
            row.append(v['type'])
            row.append(v['default'] if 'default' in v else ' ')
            row.append(v['optional'] if 'optional' in v else ' ')
            row.append(v['enum'] if 'enum' in v else ' ')

            if check_comments(v):
                rows.append(row)
                doc.add_table(headers, rows)
                doc = insert_comments_between_inputs(doc, v)
                rows = []
            else:
                rows.append(row)

        doc.add_table(headers, rows)

    return doc


def add_outputs(doc, asset):
    doc.add_heading("Outputs ", level=2)

    if "outputs" in asset:
        headers = ['Name', 'Description', 'Type']
        rows = []
        if asset.ca.items.get('outputs') is not None:
            doc = insert_comments_under_input(doc, asset.ca.items.get('outputs'))

        for k, v in asset['outputs'].items():
            row = []
            row.append(k)
            row.append(v['description'] if 'description' in v else ' ')
            row.append(v['type'])

            if check_comments(v):
                rows.append(row)
                doc.add_table(headers, rows)
                doc = insert_comments_between_inputs(doc, v)
                rows = []
            else:
                rows.append(row)

        doc.add_table(headers, rows)

    return doc