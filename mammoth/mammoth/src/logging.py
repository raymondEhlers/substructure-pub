""" Logging objects for use with c++ extensions

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from mammoth.framework import logging_utils

jet_finding_logger_stdout = logging_utils.StreamLogger(logging.getLogger("mammoth.framework.jet_finding"), logging.DEBUG)
jet_finding_logger_stderr = logging_utils.StreamLogger(logging.getLogger("mammoth.framework.jet_finding"), logging.WARNING)
