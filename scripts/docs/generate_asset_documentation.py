# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
import snakemd
import argparse
import re
import os


def create_asset_doc(spec_file_name, asset_type):
    # read the component yaml file and parse it
    yaml = YAML()
    with open(spec_file_name, 'r') as f:
        component = yaml.load(f)

    asset_type = str(asset_type).replace("AssetType.", "").lower()

    # create component document and add info
    doc = snakemd.new_doc()
    doc = add_intro(doc, component)
    doc = add_inputs(doc, component)
    doc = add_outputs(doc, component)
    doc = add_additional_details(doc, component)

    # check if asset folder exists, create one if not
    if not os.path.exists(asset_type + "s/"):
        os.mkdir(asset_type + "s")

    # write the file into the asset folder
    with open(asset_type + "s/" + asset_type + "-" + component["name"] + ".md", 'w') as f:
        f.write(str(doc))

    # return md file name
    return asset_type, asset_type + "-" + component["name"]



def add_intro(doc, component):

    doc.add_heading(component['display_name'] if "display_name" in component else component["name"], level=2)

    doc.add_heading("README file ", level=3)

    doc.add_heading("Component Overview ", level=3)
    
    if "description" in component:
        doc.add_paragraph("**Description**: " + str(component['description']))
    if "version" in component:
        doc.add_paragraph("**Version**: " + str(component['version']))
    if "type" in component:
        doc.add_paragraph("**Type**: " + component['type'])
    if "tags" in component and "license" in component["tags"]:
        doc.add_paragraph("**License**: " + component["tags"]["license"])

    doc.add_heading("YAML Syntax ", level=3)

    if "type" in component and component["type"] == "automl":
        doc.add_paragraph("**Task**: " + component["task"] if "task" in component else "")

    if "properties" in component:
        doc.add_heading("Properties", level=4)
        doc.add_paragraph("**SHA**: " + component["properties"]["SHA"] if "SHA" in component["properties"] else "")
        doc.add_paragraph("**Datasets**: " + component["properties"]["datasets"] if "datasets" in component["properties"] else "")
        doc.add_paragraph("**Finetuning Tasks**: " + component["properties"]["finetuning-tasks"] if "finetuning-tasks" in component["properties"] else "")
        doc.add_paragraph("**Languages**: " + component["properties"]["languages"] if "languages" in component["properties"] else "")

    return doc


def add_additional_details(doc, component):

    doc.add_heading("Parameters ", level=3)
    
    doc.add_heading("Code", level=3)
    if "type" in component and component["type"] == "command" and "code" in component:
        doc.add_paragraph(component['code'])

    doc.add_heading("Environment ", level=3)
    doc.add_paragraph(component['environment'] if "environment" in component else "")
        
    doc.add_heading("Compute Specifications ", level=3)
    doc.add_paragraph(component['tags']['min_inference_sku'] if "tags" in component and "min_inference_sku" in component["tags"] else "")

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
        text=comment.value
        text = re.sub(r"#", "", text)
        doc.add_paragraph(text)

    return doc

def insert_comments_under_input(doc, data):
    for comments in data:
        if comments:
            for comment in comments:
                doc.add_paragraph(comment.value.strip('#'))
    return doc

def add_inputs(doc, component):
    doc.add_heading("Inputs ", level=2)

    if "inputs" in component:
        headers = ['Name', 'Description', 'Type', 'Default', 'Optional', 'Enum']
        rows = []
        if component.ca.items.get('inputs') is not None:
            doc = insert_comments_under_input(doc, component.ca.items.get('inputs'))

        for k, v in component['inputs'].items():
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


def add_outputs(doc, component):
    doc.add_heading("Outputs ", level=2)

    if "outputs" in component:
        headers = ['Name', 'Description', 'Type']
        rows = []
        if component.ca.items.get('outputs') is not None:
            doc = insert_comments_under_input(doc, component.ca.items.get('outputs'))

        for k, v in component['outputs'].items():
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

