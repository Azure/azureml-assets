# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate markdown documentation for an asset."""
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
import snakemd
import re
import os
from pathlib import Path
from glob import glob as search


def create_asset_doc(asset_config, asset_type):
    """Generate asset markdown document with information based on its asset yaml file."""
    # read the asset yaml file and parse it
    yaml = YAML()
    with open(asset_config.spec_with_path, 'r') as f:
        asset = yaml.load(f)

    asset_type = asset_type.value

    # create asset document and add info
    if asset_type == "environment":
        doc, description, name, categories = create_environment_doc(asset, asset_config)
    elif asset_type == "component":
        doc, description, name, categories = create_component_doc(asset, asset_config)
    elif asset_type == "model":
        doc, description, name, categories = create_model_doc(asset, asset_config)

    # check if asset folder exists, create one if not
    asset_type_dir = Path(f"{asset_type}s")
    asset_type_dir.mkdir(exist_ok=True)

    asset_file_name = f"{asset_type}-{asset['name']}"
    full_asset_file_name = f"{asset_type_dir}/{asset_file_name}.md"

    # write the file into the asset folder
    with open(full_asset_file_name, 'w') as f:
        f.write(str(doc))

    return asset_type, asset["name"], asset_file_name, asset.get("description", "")


def create_environment_doc(asset, asset_config):
    """Generate environment document content."""
    doc = snakemd.new_doc()

    doc = add_intro(doc, asset)
    link = "https://ml.azure.com/registries/azureml/environments/{}/version/{}".format(asset["name"], asset["version"])
    doc.add_paragraph("**View in Studio**:  [{}]({})".format(link, link))
    # doc.add_paragraph("**View in Studio**:  <a href=\"{}\" target=\"_blank\">{}</a>".format(link, link))
    doc = add_mcr_image(doc, asset, asset_config)
    doc = add_docker_context(doc, asset, asset_config)
    doc = add_additional_details(doc, asset)

    description = None
    name = None
    categories = None
    return doc, description, name, categories


def create_component_doc(asset, asset_config):
    """Generate component document content."""
    doc = snakemd.new_doc()

    doc = add_intro(doc, asset)
    doc = add_inputs(doc, asset)
    doc = add_outputs(doc, asset)
    doc = add_additional_details(doc, asset)

    description = None
    name = None
    categories = None
    return doc, description, name, categories


def create_model_doc(asset, asset_config):
    """Generate model document content."""
    doc = snakemd.new_doc()

    doc = add_intro(doc, asset)

    doc = add_inputs(doc, asset)
    doc = add_outputs(doc, asset)
    doc = add_additional_details(doc, asset)

    description = None
    name = None
    categories = None
    return doc, description, name, categories


def add_mcr_image(doc, asset, asset_config):
    """Add MCR Image."""
    img = ":".join([str(asset_config.extra_config_as_object().get_full_image_name()), asset["version"]])
    doc.add_paragraph("**Docker image**: " + img)
    return doc


def add_docker_context(doc, asset, asset_config):
    """Add docker context content."""
    doc.add_heading("Docker build context", level=2)
    context = {}
    dockerfile = ""
    for context_file in search(str(asset_config.extra_config_as_object().context_dir_with_path) + "/*"):
        content = ""
        with open(context_file, "r") as f:
            # do not expect nested structure for now
            content = f.read()
        filename = os.path.basename(context_file)
        if filename.lower() == "dockerfile":
            dockerfile = content
        else:
            context[filename] = content

    doc.add_heading("Dockerfile", level=3)
    doc.add_code(dockerfile, lang="dockerfile")

    for name, file_content in context.items():
        doc.add_heading(name, level=3)
        filename, file_extension = os.path.splitext(name)
        doc.add_code(file_content, lang=file_extension.strip("."))
    return doc


def add_intro(doc, asset, asset_type=None):
    """Add information like asset name and description to asset doc."""
    doc.add_heading(asset['display_name'] if "display_name" in asset else asset["name"], level=2)

    # doc.add_heading("README file ", level=3)

    doc.add_heading("Overview ", level=2)

    if "description" in asset:
        doc.add_paragraph("**Description**: " + str(asset['description']))

    if "version" in asset:
        doc.add_paragraph("**Version**: " + str(asset['version']))
    if "type" in asset:
        doc.add_paragraph("**Type**: " + asset['type'])
    if "tags" in asset and "license" in asset["tags"]:
        doc.add_paragraph("**License**: " + asset["tags"]["license"])

    # doc.add_heading("YAML Syntax ", level=3)

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

    if "tags" in asset:
        doc.add_heading("Tags ", level=3)
        tags = []
        for tag, value in asset["tags"].items():
            tags.append("`{}` ".format(tag if len(str(value)) == 0 else (tag + " : " + str(value))))
        tags.sort(key=lambda x: x[0].lower())
        doc.add_raw(" ".join(tags))

    return doc


def add_additional_details(doc, asset):
    """Add information like parameters and compute specifications to asset doc."""
    # doc.add_heading("Parameters ", level=3)
    
    if "type" in asset and asset["type"] == "command" and "code" in asset:
        doc.add_heading("Code", level=3)
        doc.add_paragraph(asset['code'])

    if "environment" in asset:
        doc.add_heading("Environment", level=3)
        doc.add_paragraph(asset['environment'])

    if "tags" in asset and "min_inference_sku" in asset["tags"]:
        doc.add_heading("Compute Specifications ", level=3)
        doc.add_paragraph(asset['tags']['min_inference_sku'])

    return doc


# set attributes
def get_comments_map(self, key):
    """Get comments based on key."""
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
    """Get comments based on id."""
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
    """Check for comments in asset input."""
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
    """Insert comments between inputs."""
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
    """Insert comments under inputs."""
    for comments in data:
        if comments:
            for comment in comments:
                doc.add_paragraph(comment.value.strip('#'))
    return doc


def add_inputs(doc, asset):
    """Generate inputs table for the asset doc."""
    if "inputs" in asset:
        doc.add_heading("Inputs ", level=2)

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
    """Generate an outputs table for the asset doc."""

    if "outputs" in asset:
        doc.add_heading("Outputs ", level=2)
        
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
