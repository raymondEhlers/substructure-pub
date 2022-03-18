"""Convert jet extractor into expected awkward array format.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Any, MutableMapping, Optional, Tuple

import attr
import awkward as ak

from mammoth.framework import sources

logger = logging.getLogger(__name__)


@attr.define
class JEWELSource:
    _filename: Path = attr.field(converter=Path)
    metadata: MutableMapping[str, Any] = attr.Factory(dict)

    def __len__(self) -> int:
        """Number of entries in the source."""
        if "n_entries" in self.metadata:
            return int(self.metadata["n_entries"])
        raise ValueError("N entries not yet available.")

    def data(self) -> ak.Array:
        """Return data from the source.

        Returns:
            Data in an awkward array.
        """
        if "parquet" not in self._filename.suffix:
            arrays = jet_extractor_to_awkward(
                filename=self._filename,
                # We always want to pull in as many tracks as possible, so take the largest possible R
                jet_R=0.6,
            )
        else:
            source = sources.ParquetSource(
                filename=self._filename,
            )
            arrays = source.data()
        self.metadata["n_entries"] = len(arrays)
        return arrays


def jet_extractor_to_awkward(
    filename: Path,
    jet_R: float,
    entry_range: Optional[Tuple[int, int]] = None,
) -> ak.Array:
    # For JEWEL, these were the only meaningful columns
    event_level_columns = {
        "Event_Weight": "event_weight",
        "Event_ImpactParameter": "event_impact_parameter",
        # Hannah needs this for the extractor bin scaling
        "Jet_Pt": "jet_pt_original",
    }
    particle_columns = {
        "Jet_Track_Pt": "pt",
        "Jet_Track_Eta": "eta",
        "Jet_Track_Phi": "phi",
        "Jet_Track_Charge": "charged",
        "Jet_Track_Label": "label",
    }

    additional_uproot_source_kwargs = {}
    if entry_range is not None:
        additional_uproot_source_kwargs = {
            "entry_range": entry_range
        }

    data = sources.UprootSource(
        filename=filename,
        tree_name=f"JetTree_AliAnalysisTaskJetExtractor_JetPartLevel_AKTChargedR{round(jet_R * 100):03}_mctracks_pT0150_E_scheme_allJets",
        columns=list(event_level_columns) + list(particle_columns),
        **additional_uproot_source_kwargs,  # type: ignore
    ).data()

    return ak.Array({
        "part_level": ak.zip(
            dict(
                zip(
                    #[c.replace("Jet_T", "t").lower() for c in list(particle_columns)],
                    list(particle_columns.values()),
                    ak.unzip(data[particle_columns]),
                )
            )
        ),
        **dict(
            zip(
                list(event_level_columns.values()),
                ak.unzip(data[event_level_columns]),
            )
        ),
    })


def write_to_parquet(arrays: ak.Array, filename: Path) -> bool:
    """Write the jagged track skim arrays to parquet.

    In this form, they should be ready to analyze.
    """
    # Determine the types for improved compression when writing
    # See the notes in track_skim for why some choices are made.
    # Columns to store as integers
    dictionary_encoded_columns = [
        # NOTE: Uses notation from arrow/parquet
        #       `list.item` basically gets us to an column in the list.
        #       This may be a little brittle, but let's see.
        "part_level.list.item.label",
    ]

    # Columns to store as float
    byte_stream_split_columns = [
        "event_weight",
        "event_impact_parameter",
        "jet_pt_original",
        "part_level.list.item.pt",
        "part_level.list.item.eta",
        "part_level.list.item.phi",
        "part_level.list.item.charge",
    ]

    # NOTE: If there are issues about reading the files due to arrays being too short, check that
    #       there are no empty events. Empty events apparently cause issues for byte stream split
    #       encoding: https://issues.apache.org/jira/browse/ARROW-13024
    #       Unfortunately, this won't become clear until reading is attempted.
    ak.to_parquet(
        array=arrays,
        where=filename,
        compression="zstd",
        # Use for anything other than floats
        use_dictionary=dictionary_encoded_columns,
        # Optimize for floats for the rest
        use_byte_stream_split=byte_stream_split_columns,
    )

    return True


if __name__ == "__main__":
    import mammoth.helpers
    mammoth.helpers.setup_logging(level=logging.INFO)

    # Setup
    JEWEL_identifier = "NoToy_PbPb"
    # Since the PbPb files tend to have a few thousand or fewer events, we want the JEWEL chunk size
    # to be similar to that value. Otherwise, the start of JEWEL files will be embedded far too often,
    # while the ends will never be reached.
    # We also want to keep the num
    # ber of files in check. 5000 seems like a reasonable balance.
    chunk_size = int(5e3)

    # Map from JEWEL identifier to a somewhat clearer name for directories, etc
    JEWEL_label = {
        "NoToy_PbPb": "central_00_10",
        "NoToy_PbPb_3050": "semi_central_30_50",
    }

    for pt_hat_bin in [
        "05_15",
        "15_30",
        "30_45",
        "45_60",
        "60_80",
        "80_140",
    ]:
        filename = Path(f"/alf/data/rehlers/skims/JEWEL_PbPb_no_recoil/JEWEL_{JEWEL_identifier}_PtHard{pt_hat_bin}.root")

        # Keep track of iteration
        start = 0
        continue_iterating = True
        index = 0
        while continue_iterating:
            end = start + chunk_size
            logger.info(f"Processing file {filename}, chunk {index} from {start}-{end}")

            arrays = jet_extractor_to_awkward(
                filename=filename,
                # Use jet R = 0.6 because this will contain more of the JEWEL particles.
                # We should be safe to use this for embedding for smaller R jets too, since they
                # should be encompassed within the R = 0.6 jet.
                jet_R=0.6,
                entry_range=(start, end),
            )
            # Just for confirmation that it matches the chunk size (or is smaller)
            logger.debug(f"Array length: {len(arrays)}")

            output_dir = filename.parent / "skim" / JEWEL_label[JEWEL_identifier]
            output_dir.mkdir(parents=True, exist_ok=True)
            write_to_parquet(
                arrays=arrays,
                filename=(output_dir / f"{filename.stem}_{index:03}").with_suffix('.parquet'),
            )

            if len(arrays) < (end - start):
                # We're out of entries - we're done.
                break

            # Move up to the next iteration.
            start = end
            index += 1

        logger.info(f"Finished at index {index} for pt hat bin {pt_hat_bin}")

    #import IPython; IPython.start_ipython(user_ns={**globals(),**locals()})
