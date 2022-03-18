#!/usr/bin/env python3

""" Download file list in YAML with pachyderm

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
import queue
from typing import List

from pachyderm import yaml
from pachyderm.alice import download

from jet_substructure.base import helpers


logger = logging.getLogger(__name__)


def create_file_pairs() -> List[download.FilePair]:
    """Create file pairs.

    YAML file is of the form:
    alien_file: local_file
    """
    y = yaml.yaml()
    with open("files_to_download.yaml", "r") as f:
        file_list_input = y.load(f)

    file_pairs = []
    for alien_file, local_file in file_list_input.items():
        file_pairs.append(download.FilePair(source=alien_file, target=local_file))

    logger.debug(f"file_pairs: {file_pairs}")

    return file_pairs


if __name__ == "__main__":
    helpers.setup_logging(level=logging.INFO)

    file_pairs = create_file_pairs()

    # Setup the queue and filler, and then start downloading.
    q: download.FilePairQueue = queue.Queue()
    queue_filler = download.FileListDownloadFiller(pairs=file_pairs, q=q)
    download.download(queue_filler=queue_filler, q=q)
