# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate markdown documentation for an asset."""
from typing import List
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
import snakemd
import re
import os
from pathlib import Path
from azureml.assets.config import AssetConfig, AssetType
from glob import glob as search

SUPPORTED_ASSET_TYPES = [AssetType.ENVIRONMENT, AssetType.COMPONENT, AssetType.MODEL]
DEFAULT_CATEGORY = "Uncategorized"


class AssetInfo:
    """Base class for Assets documentation."""

    _asset_config = None
    _asset = None
    _extra_config_object = None
    _doc = None

    def __init__(self, asset_config: AssetConfig):
        """Initialize content."""
        self._asset_config = asset_config
        yaml = YAML()
        with open(self._asset_config.spec_with_path, 'r') as f:
            self._asset = yaml.load(f)
        self._extra_config_object = self._asset_config.extra_config_as_object()

    @property
    def type(self) -> str:
        """Return asset type."""
        return self._asset_config.type.value

    @property
    def name(self) -> str:
        """Return asset name."""
        return self._asset["name"]

    @property
    def version(self) -> str:
        """Return asset version."""
        return str(self._asset["version"])

    @property
    def description(self) -> str:
        """Return asset description."""
        return str(self._asset.get("description", ""))

    @property
    def categories(self) -> List[str]:
        """Return asset categories."""
        return [DEFAULT_CATEGORY]

    @property
    def filename(self) -> str:
        """Return asset filename."""
        return f"{self.type}-{self.name}"

    @property
    def directory(self) -> Path:
        """Return asset directory."""
        return Path(f"{self.type}s")

    @property
    def fullpath(self) -> str:
        """Return asset document full path."""
        return f"{self.directory}/{self.filename}.md"

    @property
    def doc(self) -> snakemd.document.Document:
        """Generate markdown document."""
        raise Exception("Not implemented")

    def save(self):
        """Save markdown document."""
        self.directory.mkdir(exist_ok=True)
        with open(self.fullpath, 'w') as f:
            f.write(str(self.doc))

    @staticmethod
    def create_asset_info(asset_config):
        """Instantiate an asset info class."""
        # TODO: Use AssetType.COMPONENT
        if asset_config.type == AssetType.ENVIRONMENT:
            return EnvironmentInfo(asset_config)
        if asset_config.type == AssetType.COMPONENT:
            return ComponentInfo(asset_config)
        if asset_config.type == AssetType.MODEL:
            return ModelInfo(asset_config)

        raise Exception(f"Not supported asset type {asset_config.type}. Use {SUPPORTED_ASSET_TYPES}")

# region Doc Formatting
    def _add_doc_name(self, doc):
        if "display_name" in self._asset:
            doc.add_heading(self._asset['display_name'])
        doc.add_heading(self.name)

    def _add_doc_overview(self, doc):
        doc.add_heading("Overview ", level=2)

    def _add_doc_description(self, doc):
        doc.add_paragraph("**Description**: " + self.description)

    def _add_doc_asset_version(self, doc):
        doc.add_paragraph("**Version**: " + self.version)

    def _add_doc_license_from_tags(self, doc):
        if "tags" in self._asset and "license" in self._asset["tags"]:
            doc.add_paragraph("**License**: " + self._asset["tags"]["license"])

    def _add_doc_properties(self, doc):
        if "properties" in self._asset:
            doc.add_heading("Properties", level=4)
            for property, value in self._asset["properties"].items():
                doc.add_paragraph(f"**{property}**: {value}")

    def _add_doc_tags(self, doc):
        if "tags" in self._asset:
            doc.add_heading("Tags ", level=3)
            tags = []
            for tag, value in self._asset["tags"].items():
                tags.append("`{}` ".format(tag if len(str(value)) == 0 else (tag + " : " + str(value))))
            tags.sort(key=lambda x: x[0].lower())
            doc.add_raw(" ".join(tags))
        return doc

    def _add_doc_mcr_image(self, doc):
        """Add MCR Image."""
        img = ":".join([self._extra_config_object.get_full_image_name(), self.version])
        doc.add_paragraph("**Docker image**: " + img)

    def _add_doc_docker_context(self, doc):
        """Add docker context content."""
        doc.add_heading("Docker build context", level=2)
        context = {}
        dockerfile = ""
        for context_file in search(str(self._extra_config_object.context_dir_with_path) + "/*"):
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

    def _add_doc_component_type_and_code(self, doc):
        # TODO: have a code reference on GH
        # doc.add_heading("Code", level=3)
        # doc.add_paragraph(self._asset['code'])
        pass

    def _add_doc_environment_dependency(self, doc):
        if "environment" in self._asset:
            doc.add_heading("Environment", level=3)
            doc.add_paragraph(self._asset['environment'])

    def _add_doc_compute_sku(self, doc):
        if "tags" in self._asset and "min_inference_sku" in self._asset["tags"]:
            doc.add_heading("Compute Specifications ", level=3)
            doc.add_paragraph(self._asset['tags']['min_inference_sku'])

    @staticmethod
    def _add_doc_insert_comments_between_inputs(doc, data):
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

    @staticmethod
    def _add_doc_insert_comments_under_input(doc, data):
        """Insert comments under inputs."""
        for comments in data:
            if comments:
                for comment in comments:
                    doc.add_paragraph(comment.value.strip('#'))

    def _add_doc_asset_inputs(self, doc):
        """Generate inputs table for the asset doc."""
        if "inputs" in self._asset:
            doc.add_heading("Inputs ", level=2)

            headers = ['Name', 'Description', 'Type', 'Default', 'Optional', 'Enum']
            rows = []
            if self._asset.ca.items.get('inputs') is not None:
                self._add_doc_insert_comments_under_input(doc, self._asset.ca.items.get('inputs'))

            for k, v in self._asset['inputs'].items():
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
                    self._add_doc_insert_comments_between_inputs(doc, v)
                    rows = []
                else:
                    rows.append(row)

            doc.add_table(headers, rows)

    def _add_doc_asset_outputs(self, doc):
        """Generate an outputs table for the asset doc."""
        if "outputs" in self._asset:
            doc.add_heading("Outputs ", level=2)

            headers = ['Name', 'Description', 'Type']
            rows = []
            if self._asset.ca.items.get('outputs') is not None:
                doc = self._add_doc_insert_comments_under_input(doc, self._asset.ca.items.get('outputs'))

            for k, v in self._asset['outputs'].items():
                row = []
                row.append(k)
                row.append(v['description'] if 'description' in v else ' ')
                row.append(v['type'])

                if check_comments(v):
                    rows.append(row)
                    doc.add_table(headers, rows)
                    self._add_doc_insert_comments_between_inputs(doc, v)
                    rows = []
                else:
                    rows.append(row)

            doc.add_table(headers, rows)
# endregion


class EnvironmentInfo(AssetInfo):
    """Environment asset class."""

    def __init__(self, asset_config: AssetConfig):
        """Instantiate Environment asset class."""
        super().__init__(asset_config)

    @AssetInfo.doc.getter
    def doc(self) -> snakemd.document.Document:
        """Generate environment markdown document."""
        _doc = snakemd.new_doc()

        self._add_doc_name(_doc)
        self._add_doc_overview(_doc)
        self._add_doc_description(_doc)
        self._add_doc_asset_version(_doc)
        self._add_doc_tags(_doc)

        link = "https://ml.azure.com/registries/azureml/environments/{}/version/{}".format(self.name, self.version)
        _doc.add_paragraph("**View in Studio**:  [{}]({})".format(link, link))
        # doc.add_paragraph("**View in Studio**:  <a href=\"{}\" target=\"_blank\">{}</a>".format(link, link))

        self._add_doc_mcr_image(_doc)
        self._add_doc_docker_context(_doc)

        return _doc


class ComponentInfo(AssetInfo):
    """Component asset class."""

    def __init__(self, asset_config: AssetConfig):
        """Instantiate Component asset class."""
        super().__init__(asset_config)

    @AssetInfo.doc.getter
    def doc(self) -> snakemd.document.Document:
        """Generate component markdown document."""
        _doc = snakemd.new_doc()

        self._add_doc_name(_doc)
        self._add_doc_overview(_doc)
        self._add_doc_description(_doc)
        self._add_doc_asset_version(_doc)
        self._add_doc_tags(_doc)

        self._add_doc_asset_inputs(_doc)
        self._add_doc_asset_outputs(_doc)

        self._add_doc_component_type_and_code(_doc)
        self._add_doc_environment_dependency(_doc)

        return _doc


class ModelInfo(AssetInfo):
    """Model asset class."""

    def __init__(self, asset_config: AssetConfig):
        """Instantiate Model asset class."""
        super().__init__(asset_config)

    @AssetInfo.doc.getter
    def doc(self) -> snakemd.document.Document:
        """Generate model markdown document."""
        _doc = snakemd.new_doc()
        self._add_doc_name(_doc)
        self._add_doc_overview(_doc)
        self._add_doc_description(_doc)
        self._add_doc_asset_version(_doc)
        self._add_doc_tags(_doc)
        self._add_doc_license_from_tags(_doc)
        self._add_doc_properties(_doc)
        self._add_doc_compute_sku(_doc)

        return _doc


class Categories:
    """Categories structured by type"""
    _categories = {}

    def __init__(self):
        self._categories[AssetType.ENVIRONMENT.value] = {}
        self._categories[AssetType.COMPONENT.value] = {}
        self._categories[AssetType.MODEL.value] = {}

    def classify_asset(self, asset: AssetInfo):
        for category in asset.categories:
            cats = category.split("/")
            top = cats[0]
            if top not in self._categories[asset.type]:
                self._categories[asset.type][top] = CategoryInfo(top)
            self._categories[asset.type][top].add_asset(asset, cats[1:])

    # save should be category based
    def save(self):
        for type, category in self._categories.items():
            # Create a new markdown file for each asset type
            doc = snakemd.new_doc()
            doc.add_heading(type.capitalize() + "s", level=1)

            # Create glossary that links to each asset of the asset type
            doc.add_heading("Glossary", level=2)

            doc.add_horizontal_rule()

            for asset in category[DEFAULT_CATEGORY].assets:
                doc.add_unordered_list([snakemd.Paragraph(asset.name).insert_link(asset.name, asset.filename)])
                # limit description to 300 chars
                description = asset.description if len(asset.description) < 300 else (asset.description[:297] + "...")
                doc.add_raw("\n  > " + description)

            with open(f"{type}s/{type}s-documentation.md", 'w') as f:
                f.write(str(doc))        


class CategoryInfo:
    """Category class"""
    _name = None
    _assets = []
    _categories = {}
    _parent = None

    def __init__(self, name: str, parent=None):
        self._name = name
        self._parent = parent

    def add_asset(self, asset: AssetInfo, sub_categories:List[str]):
        if sub_categories:
            top = sub_categories[0]
            if top not in self._categories:
                self._categories[top] = CategoryInfo(top, self)
            self._categories[top].add_asset(asset, sub_categories[1:])
        # Add assets to all parent categories
        self._assets.append(asset)

    @property
    def assets(self):
        return self._assets

# region attibutes
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

# endregion
