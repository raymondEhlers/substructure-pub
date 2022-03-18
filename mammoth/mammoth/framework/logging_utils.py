"""Logger utilities

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from typing import Any, List


class StreamLogger:
    """Logger that can be used in place of a stream.

    From: https://stackoverflow.com/a/66209331/12907985
    """

    def __init__(self, logger: Any, level: int):
        self.logger = logger
        self.log_level = level
        self.buf: List[str] = []

    def write(self, msg: str) -> None:
        if msg.endswith("\n"):
            # Python 3.9+
            # self.buf.append(msg.removesuffix('\n'))
            # Before python 3.9
            self.buf.append(msg.rstrip("\n"))
            self.logger.log(self.log_level, "".join(self.buf))
            self.buf = []
        else:
            self.buf.append(msg)

    def flush(self) -> None:
        pass
