<<<<<<< HEAD
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from conversation import Conversation, Role, TextMessage, MultimodalMessage, MultimodalContent


class TestConversationModel(unittest.TestCase):
    def test_empty_conversation(self):
        # Create an empty conversation and serialize to JSON
        conv = Conversation()
        json_str = conv.to_json()
        self.assertEqual(json_str, "[]")

        # Deserialize back to an empty Conversation object
        new_conv = Conversation.from_json(json_str)
        self.assertEqual(len(new_conv.messages), 0)

    def test_first_message_validation(self):
        with self.assertRaises(ValueError):
            Conversation([TextMessage(Role.ASSISTANT, "This should fail.")])

        try:
            Conversation([TextMessage(Role.SYSTEM, "This is valid.")])
        except ValueError:
            self.fail("Conversation initialization failed when it should have succeeded.")

        try:
            Conversation([TextMessage(Role.USER, "This is valid.")])
        except ValueError:
            self.fail("Conversation initialization failed when it should have succeeded.")

    def test_add_message_validation(self):
        # Create empty conversation
        conv = Conversation()

        with self.assertRaises(ValueError):
            conv.add_message(TextMessage(Role.ASSISTANT, "This should fail."))

        conv.add_message(TextMessage(Role.SYSTEM, "This is valid."))

        try:
            conv.add_message(TextMessage(Role.USER, "This is valid."))
            conv.add_message(TextMessage(Role.ASSISTANT, "This is valid."))
        except ValueError:
            self.fail("add_message failed when it should have succeeded.")

    def test_serialization_deserialization(self):
        # Create a conversation and serialize it
        conv = Conversation([
            TextMessage(Role.SYSTEM, "You are a helpful assistant."),
            TextMessage(Role.USER, "Who won the world series in 2020?"),
            TextMessage(Role.ASSISTANT, "The Los Angeles Dodgers won the World Series in 2020."),
        ])
        json_str = conv.to_json()

        # Deserialize back to a Conversation object
        new_conv = Conversation.from_json(json_str)

        # Check if the deserialized conversation is the same as the original
        for orig_msg, new_msg in zip(conv.messages, new_conv.messages):
            self.assertEqual(orig_msg.role, new_msg.role)
            self.assertEqual(orig_msg.content, new_msg.content)

    def test_multimodal_message(self):
        multimodal_msg = MultimodalMessage(
            role=Role.USER,
            content=[
                MultimodalContent(type="text", text="Show me an image of a cat"),
                MultimodalContent(type="image_url", image_url={"url": "https://example.com/cat.jpg"}),
            ],
        )

        conv = Conversation([TextMessage(Role.SYSTEM, "I can show images."), multimodal_msg])
        json_str = conv.to_json()
        new_conv = Conversation.from_json(json_str)

        self.assertEqual(len(new_conv.messages), 2)
        self.assertIsInstance(new_conv.messages[1], MultimodalMessage)
        self.assertEqual(new_conv.messages[1].content[0].text, "Show me an image of a cat")
        self.assertEqual(new_conv.messages[1].content[1].image_url["url"], "https://example.com/cat.jpg")


if __name__ == '__main__':
    unittest.main()
=======
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
from conversation import Conversation, Role, TextMessage, MultimodalMessage, MultimodalContent


class TestConversationModel(unittest.TestCase):
    def test_empty_conversation(self):
        # Create an empty conversation and serialize to JSON
        conv = Conversation()
        json_str = conv.to_json()
        self.assertEqual(json_str, "[]")

        # Deserialize back to an empty Conversation object
        new_conv = Conversation.from_json(json_str)
        self.assertEqual(len(new_conv.messages), 0)

    def test_first_message_validation(self):
        with self.assertRaises(ValueError):
            Conversation([TextMessage(Role.ASSISTANT, "This should fail.")])

        try:
            Conversation([TextMessage(Role.SYSTEM, "This is valid.")])
        except ValueError:
            self.fail("Conversation initialization failed when it should have succeeded.")

        try:
            Conversation([TextMessage(Role.USER, "This is valid.")])
        except ValueError:
            self.fail("Conversation initialization failed when it should have succeeded.")

    def test_add_message_validation(self):
        # Create empty conversation
        conv = Conversation()

        with self.assertRaises(ValueError):
            conv.add_message(TextMessage(Role.ASSISTANT, "This should fail."))

        conv.add_message(TextMessage(Role.SYSTEM, "This is valid."))

        try:
            conv.add_message(TextMessage(Role.USER, "This is valid."))
            conv.add_message(TextMessage(Role.ASSISTANT, "This is valid."))
        except ValueError:
            self.fail("add_message failed when it should have succeeded.")

    def test_serialization_deserialization(self):
        # Create a conversation and serialize it
        conv = Conversation([
            TextMessage(Role.SYSTEM, "You are a helpful assistant."),
            TextMessage(Role.USER, "Who won the world series in 2020?"),
            TextMessage(Role.ASSISTANT, "The Los Angeles Dodgers won the World Series in 2020."),
        ])
        json_str = conv.to_json()

        # Deserialize back to a Conversation object
        new_conv = Conversation.from_json(json_str)

        # Check if the deserialized conversation is the same as the original
        for orig_msg, new_msg in zip(conv.messages, new_conv.messages):
            self.assertEqual(orig_msg.role, new_msg.role)
            self.assertEqual(orig_msg.content, new_msg.content)

    def test_multimodal_message(self):
        multimodal_msg = MultimodalMessage(
            role=Role.USER,
            content=[
                MultimodalContent(type="text", text="Show me an image of a cat"),
                MultimodalContent(type="image_url", image_url={"url": "https://example.com/cat.jpg"}),
            ],
        )

        conv = Conversation([TextMessage(Role.SYSTEM, "I can show images."), multimodal_msg])
        json_str = conv.to_json()
        new_conv = Conversation.from_json(json_str)

        self.assertEqual(len(new_conv.messages), 2)
        self.assertIsInstance(new_conv.messages[1], MultimodalMessage)
        self.assertEqual(new_conv.messages[1].content[0].text, "Show me an image of a cat")
        self.assertEqual(new_conv.messages[1].content[1].image_url["url"], "https://example.com/cat.jpg")


if __name__ == '__main__':
    unittest.main()
>>>>>>> 4736c86812f5a79482f8001ee49abe9393309f85
