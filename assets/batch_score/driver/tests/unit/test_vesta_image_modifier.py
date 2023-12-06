import os
from pathlib import Path
from urllib.parse import urlparse

import pytest
from mock import MagicMock

from src.batch_score.common.request_modification.modifiers.vesta_image_encoder import (
    FolderNotMounted,
    ImageEncoder,
    UnsuccessfulUrlResponse,
    VestaImageModificationException,
)
from src.batch_score.common.request_modification.modifiers.vesta_image_modifier import (
    VestaImageModifier,
)
from tests.fixtures.vesta_image_modifier import (
    MOCKED_BINARY_FROM_FILE,
    MOCKED_BINARY_FROM_URL,
)


def test_vesta_payload_type():
    assert VestaImageModifier.vesta_payload_type(request_obj={"transcript": []}) == "transcript"
    assert VestaImageModifier.vesta_payload_type(request_obj={"prompt": []}) == "prompt"
    assert VestaImageModifier.vesta_payload_type(request_obj={"foobar123": []}) == None
    assert VestaImageModifier.vesta_payload_type(request_obj={"prompt": [], "transcript": []}) == None

def test_is_vesta_payload():
    assert VestaImageModifier.is_vesta_payload(request_obj={"prompt": [{"type": "text", "data": "foobar"}]}) == True
    assert VestaImageModifier.is_vesta_payload(request_obj={"transcript": [{"type": "text", "data": "foobar"}]}) == True
    assert VestaImageModifier.is_vesta_payload(request_obj={"transcript": [{"type": "text", "data": "foobar"}, {"hello": "world"}]}) == False
    assert VestaImageModifier.is_vesta_payload(request_obj={"transcript": [{"hello": "world"}]}) == False
    assert VestaImageModifier.is_vesta_payload(request_obj={"prompt": [{"hello": "world"}]}) == False
    assert VestaImageModifier.is_vesta_payload(request_obj={"invalid": [{"hello": "world"}]}) == False

def test_modify(mock_get_logger, make_vesta_image_modifier, mock__b64_from_url):
    vesta_request_obj = {
    "transcript":[{
        "type": "text",
        "data": "This is just a test. These aren't even images"
    },{
        "type": "image",
        "data": "ImageUrl!https://fake.url"
    },{
        "type": "image",
        "data": "ImageUrl!https://another.fake.url"
    },{
        "type": "image",
        "data": "/9j/4AAQSkZJRgABAQAAAQABAAD/"
    },{
        "type": "image_hr",
        "data": "/9j/4AAQSkZJRgABAQAAAQABAAD/"
    },{
        "type": "text",
        "data": "End of the test"
    }]}
    vesta_image_modifier: VestaImageModifier = make_vesta_image_modifier()
    
    modified_request_obj = vesta_image_modifier.modify(request_obj=vesta_request_obj)
    
    # Assert the two URLs were called
    assert any("fake.url" in urlparse(url).hostname for url in mock__b64_from_url)
    assert any("another.fake.url" in urlparse(url).hostname for url in mock__b64_from_url)

    # Assert modifications are correct
    assert modified_request_obj["transcript"][1]["data"] == MOCKED_BINARY_FROM_URL
    assert modified_request_obj["transcript"][2]["data"] == MOCKED_BINARY_FROM_URL
    assert modified_request_obj["transcript"][3]["data"] == "/9j/4AAQSkZJRgABAQAAAQABAAD/"
    assert modified_request_obj["transcript"][4]["data"] == "/9j/4AAQSkZJRgABAQAAAQABAAD/"

def test_modify_invalid_image(mock_get_logger, make_vesta_image_modifier, mock_encode_b64):
    vesta_request_obj = {"transcript": [{"type": "image", "data": "invalid_data"}]}
    vesta_image_modifier: VestaImageModifier = make_vesta_image_modifier()

    mock_encode_b64["exception"] = FolderNotMounted()
    with pytest.raises(VestaImageModificationException):
        vesta_image_modifier.modify(vesta_request_obj)
        
    mock_encode_b64["exception"] = Exception()
    with pytest.raises(VestaImageModificationException):
        vesta_image_modifier.modify(vesta_request_obj)

def test_encode_b64_file(mock_get_logger, make_image_encoder, mock__b64_from_file):
    path_prefix = "E:/fake/folder/"
    image_encoder: ImageEncoder = make_image_encoder(image_input_folder_str = path_prefix)

    image_target = "target.png"
    image_data = f"ImageFile!{image_target}"
    encoding = image_encoder.encode_b64(image_data=image_data)
    
    assert encoding == MOCKED_BINARY_FROM_FILE
    assert mock__b64_from_file[-1] == os.path.join(str(Path(path_prefix)), str(Path(image_target)))
    assert mock_get_logger.debug.called or mock_get_logger.info.called

    image_encoder: ImageEncoder = make_image_encoder(image_input_folder_str = None)
    with pytest.raises(FolderNotMounted):
        image_encoder.encode_b64(image_data=image_data)

def test_encode_b64_url(mock_get_logger, make_image_encoder, mock__b64_from_url):
    image_encoder: ImageEncoder = make_image_encoder(image_input_folder_str = None)

    image_target = "https://fake.url"
    image_data = f"ImageUrl!{image_target}"
    encoding = image_encoder.encode_b64(image_data=image_data)
    
    assert encoding == MOCKED_BINARY_FROM_URL
    assert mock__b64_from_url[-1] == image_target
    assert mock_get_logger.debug.called or mock_get_logger.info.called

def test__b64_from_url(mock_get_logger: MagicMock, monkeypatch, make_image_encoder):
    image_encoder: ImageEncoder = make_image_encoder()
    target_url = "https://fake.url"

    mock_get = MagicMock()
    mock_get.status_code = 200
    mock_get.content = b"MOCK_CONTENT_BYTES"
    monkeypatch.setattr("requests.get", lambda url: mock_get)

    encoding = image_encoder._b64_from_url(target_url)
    assert encoding

    mock_get.status_code = 404
    mock_get.content = None
    mock_get.reason = "mocked for testing purposes"
    monkeypatch.setattr("requests.get", lambda url: mock_get)

    with pytest.raises(UnsuccessfulUrlResponse):
        image_encoder._b64_from_url(target_url)
    assert mock_get_logger.info.called
    assert target_url in mock_get_logger.info.call_args.args[0]

def test_encode_b64_raw(mock_get_logger, make_image_encoder):
    image_encoder: ImageEncoder = make_image_encoder(image_input_folder_str = None)
    image_data = "MOCK_DATA"
    encoding = image_encoder.encode_b64(image_data=image_data)

    assert encoding == image_data
    assert mock_get_logger.debug.called or mock_get_logger.info.called