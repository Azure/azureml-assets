# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Generate markdown documentation for an asset."""
from typing import List
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
import snakemd
import re
from pathlib import Path
from azureml.assets.config import AssetConfig, AssetType
import warnings

SUPPORTED_ASSET_TYPES = [AssetType.ENVIRONMENT, AssetType.COMPONENT, AssetType.MODEL, AssetType.DATA]
DEFAULT_CATEGORY = "Uncategorized"
PLURALIZED_ASSET_TYPE = {"environment": "environments", "component": "components", "model": "models", "data": "data",
                         "prompt": "prompts"}


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
    def pluralized_type(self) -> str:
        """Return pluralized asset type."""
        return PLURALIZED_ASSET_TYPE[self._asset_config.type.value]

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
        if self._asset_config.categories:
            return ["/".join([DEFAULT_CATEGORY, cat]) for cat in self._asset_config.categories]
        else:
            return [DEFAULT_CATEGORY]

    @property
    def filename(self) -> str:
        """Return asset filename."""
        return f"{self.pluralized_type}-{self.name}"

    @property
    def directory(self) -> Path:
        """Return asset directory."""
        return Path(f"{self.pluralized_type}")

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
        if asset_config.type not in TYPE_TO_DOC_FUNCS:
            warnings.warn(f"Not supported asset type {asset_config.type}. Use {SUPPORTED_ASSET_TYPES}")
            return None

        return TYPE_TO_DOC_FUNCS[asset_config.type](asset_config)

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

    def _add_doc_link(self, doc):
        link = "https://ml.azure.com/registries/azureml/{}/{}/version/{}".format(
            self.pluralized_type, self.name, self.version)
        doc.add_paragraph("**View in Studio**:  [{}]({})".format(link, link))
        # doc.add_paragraph("**View in Studio**:  <a href=\"{}\" target=\"_blank\">{}</a>".format(link, link))

    def _add_doc_mcr_image(self, doc):
        """Add MCR Image."""
        img = ":".join([self._extra_config_object.get_full_image_name(), self.version])
        doc.add_paragraph("**Docker image**: " + img)

    def _add_doc_docker_context(self, doc):
        """Add docker context content."""
        doc.add_heading("Docker build context", level=2)

        doc.add_heading("Dockerfile", level=3)
        dockerfile = self._extra_config_object.get_dockerfile_contents()
        doc.add_code(dockerfile, lang="dockerfile")

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
                self._add_doc_insert_comments_under_input(doc, self._asset.ca.items.get('outputs'))

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
        self._add_doc_link(_doc)

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
        self._add_doc_link(_doc)

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
        self._add_doc_link(_doc)

        self._add_doc_license_from_tags(_doc)
        self._add_doc_properties(_doc)
        self._add_doc_compute_sku(_doc)

        return _doc


class DataInfo(AssetInfo):
    """Data asset class."""

    def __init__(self, asset_config: AssetConfig):
        """Instantiate Data asset class."""
        super().__init__(asset_config)

    @AssetInfo.doc.getter
    def doc(self) -> snakemd.document.Document:
        """Generate data markdown document."""
        _doc = snakemd.new_doc()
        self._add_doc_name(_doc)
        self._add_doc_overview(_doc)
        self._add_doc_description(_doc)
        self._add_doc_asset_version(_doc)
        self._add_doc_tags(_doc)
        self._add_doc_link(_doc)

        return _doc


class PromptInfo(AssetInfo):
    """Prompt asset class."""

    def __init__(self, asset_config: AssetConfig):
        """Instantiate Prompt asset class."""
        super().__init__(asset_config)

    @AssetInfo.doc.getter
    def doc(self) -> snakemd.document.Document:
        """Generate prompt markdown document."""
        _doc = snakemd.new_doc()
        self._add_doc_name(_doc)
        self._add_doc_overview(_doc)
        self._add_doc_description(_doc)
        self._add_doc_asset_version(_doc)
        self._add_doc_tags(_doc)

        return _doc


TYPE_TO_DOC_FUNCS = {
    AssetType.ENVIRONMENT: EnvironmentInfo,
    AssetType.COMPONENT: ComponentInfo,
    AssetType.MODEL: ModelInfo,
    AssetType.DATA: DataInfo
}


class Categories:
    """Categories structured by type."""

    _categories = {}

    def __init__(self):
        """Instantiate root categories."""
        self._categories[AssetType.ENVIRONMENT.value] = None
        self._categories[AssetType.COMPONENT.value] = None
        self._categories[AssetType.MODEL.value] = None
        self._categories[AssetType.DATA.value] = None
        self._categories[AssetType.PROMPT.value] = None

    def classify_asset(self, asset: AssetInfo):
        """Classify an asset."""
        for category in asset.categories:
            cats = category.split("/")
            top = cats[0]
            if self._categories[asset.type] is None:
                self._categories[asset.type] = CategoryInfo(top, asset.type)
            self._categories[asset.type].add_asset(asset, cats[1:])

    # save should be category based
    def save(self):
        """Save category documents."""
        for type, category in self._categories.items():
            if category:
                category.save()


class CategoryInfo:
    """Category class."""

    def __init__(self, name: str, type: str, parent=None):
        """Instantiate category."""
        self._name = name
        self._parent = parent
        self._type = type
        self._pluralized_type = PLURALIZED_ASSET_TYPE[type]
        self._assets = []
        self._sub_categories = {}

    @property
    def _category_full_path(self):
        parent_category = self._parent._category_full_path if self._parent else None
        return (parent_category + "-" + self._name) if parent_category else (self._pluralized_type)

    @property
    def _doc_name(self):
        return f"{self._category_full_path}-documentation".replace(" ", "-")

    @property
    def _doc_full_path_name(self):
        return f"{self._pluralized_type}/{self._doc_name}.md"

    @property
    def _is_root(self):
        return self._name == DEFAULT_CATEGORY

    def add_asset(self, asset: AssetInfo, sub_categories: List[str]):
        """Add an asset to category."""
        if sub_categories:
            top = sub_categories[0]
            if top not in self._sub_categories:
                self._sub_categories[top] = CategoryInfo(top, self._type, self)
            self._sub_categories[top].add_asset(asset, sub_categories[1:])
        # Add assets to all parent categories
        if asset not in self._assets:
            self._assets.append(asset)

    @property
    def assets(self):
        """Assets."""
        return self._assets

    def save(self):
        """Save category documents."""
        doc = snakemd.new_doc()
        if self._is_root:
            doc.add_heading(self._pluralized_type.capitalize(), level=1)
        else:
            doc.add_heading(self._name, level=1)

        if self._sub_categories:
            sorted_sub_categories = dict(sorted(self._sub_categories.items(), key=lambda i: i[0].lower()))
            doc.add_heading("Categories", level=2)
            for name, child in sorted_sub_categories.items():
                child.save()
                doc.add_unordered_list([snakemd.Paragraph(child._name).insert_link(child._name, child._doc_name)])

        # Create glossary that links to each asset of the asset type
        if self._is_root:
            doc.add_heading(f"All {self._pluralized_type}", level=2)
        else:
            doc.add_heading(f"{self._pluralized_type.capitalize()} in this category", level=2)

        doc.add_horizontal_rule()

        self.assets.sort(key=lambda x: x.name.lower())

        for asset in self.assets:
            doc.add_unordered_list([snakemd.Paragraph(asset.name).insert_link(asset.name, asset.filename)])
            # limit description to 300 chars
            description = asset.description if len(asset.description) < 300 else (asset.description[:297] + "...")
            doc.add_raw("\n  > " + description)

        with open(self._doc_full_path_name, 'w') as f:
            f.write(str(doc))


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
