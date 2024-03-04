# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

""" Cancel handler class. """

import signal


class CancelHandler:
    """Cancel handler"""
    cancel_triggered = False

    def __init__(self):
        """ Constructor for CancelHandler class. """
        signal.signal(signal.SIGINT, self.cancel)
        signal.signal(signal.SIGTERM, self.cancel)

    def cancel(self, *args):
        """ Cancel method to handle the signal."""
        print('cancel triggered')
        self.cancel_triggered = True
