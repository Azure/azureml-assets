# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Functions for replacing template tags."""


TAG_PREFIX = "{{"
TAG_SUFFIX = "}}"
TAG_SEPARATOR = "."


def _process_tag(tag: str, data: dict) -> str:
    """Replace tag with data value, or leave intact if not found in data.

    Args:
        tag (str): Template tag, including prefix and suffix.
        data (dict): Data to replace tag with, may include nested dictionaries.

    Returns:
        str: The data value or original template tag if not found.
    """
    inside_tag = tag[len(TAG_PREFIX):-len(TAG_SUFFIX)].strip()
    tag_segments = inside_tag.split(TAG_SEPARATOR)
    data_context = data
    for tag_segment in tag_segments:
        val = data_context.get(tag_segment)
        if isinstance(val, dict):
            data_context = val
        elif val is not None:
            return val
        else:
            return tag


def render(template: str, data: dict) -> str:
    """Replace template tags with data, or leave them intact if not found in data.

    Args:
        template (str): Template that may include tags to replace.
        data (dict): Data to replace tags with, may include nested dictionaries.

    Returns:
        str: Template with resolved tags replaced with their values.
    """
    # Split template into a list of strings separated by template tags
    template_components = []
    start = 0
    while start < len(template) - 1:
        # Find a template tag
        tag_start = template.find(TAG_PREFIX, start)
        tag_suffix_start = template.find(TAG_SUFFIX, tag_start) if tag_start != -1 else -1
        if tag_start == -1 or tag_suffix_start == -1:
            # No template tag found, store remainder
            template_components.append(template[start:])
            break

        # Compute end position of tag
        tag_end = tag_suffix_start + len(TAG_SUFFIX) - 1

        if tag_start > start:
            # Store string before the template tag
            template_components.append(template[start:tag_start])
        # Store template tag
        template_components.append(template[tag_start:tag_end + 1])
        start = tag_end + 1

    # Process template tags
    for i, template_component in enumerate(template_components):
        # Skip non-tags
        if not template_component.startswith(TAG_PREFIX):
            continue

        # Replace tag with data value or leave intact
        template_components[i] = _process_tag(template_component, data)

    return "".join(template_components)
