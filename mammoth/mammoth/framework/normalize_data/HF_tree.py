"""Convert HF tree to parquet, making it easier to use.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from pathlib import Path

import awkward as ak
import numpy as np

from mammoth.framework import sources, utils


def hf_tree_to_awkward_MC(
    filename: Path,
    collision_system: str,
) -> ak.Array:
    # TODO: Consolidate with the _data function
    # Setup
    # First, we need the identifiers to group_by
    # According to James:
    # Both data and MC need run_number and ev_id.
    # Data additionally needs ev_id_ext
    identifiers = [
        "run_number",
        "ev_id",
    ]
    if collision_system in ["pp", "PbPb"]:
        identifiers += ["ev_id_ext"]
    # Particle columns and names.
    particle_level_columns = identifiers + [
        "ParticlePt",
        "ParticleEta",
        "ParticlePhi",
    ]
    # We want them to be stored in a standardized manner.
    _standardized_particle_names = {
        "ParticlePt": "pt",
        "ParticleEta": "eta",
        "ParticlePhi": "phi",
    }

    # Detector level
    det_level_tracks_source = sources.UprootSource(
        filename=filename,
        tree_name="PWGHF_TreeCreator/tree_Particle",
        columns=particle_level_columns,
    )
    # Particle level
    part_level_tracks_source = sources.UprootSource(
        filename=filename,
        tree_name="PWGHF_TreeCreator/tree_Particle_gen",
        columns=particle_level_columns,
    )
    # Event level properties
    event_properties_columns = ["z_vtx_reco", "is_ev_rej"]
    # Collision system customization
    if collision_system == "PbPb":
        event_properties_columns += ["centrality"]
        # For the future, perhaps can add:
        # - event plane angle (but doesn't seem to be in HF tree output :-( )
    # It seems that the pythia relevant properties like pt hard bin, etc, are
    # all empty so nothing special to be done there.
    event_properties_source = sources.UprootSource(
        filename=filename,
        tree_name="PWGHF_TreeCreator/tree_event_char",
        columns=identifiers + event_properties_columns,
    )

    # Convert the flat arrays into jagged arrays by grouping by the identifiers.
    # This allows us to work with the data as expected.
    det_level_tracks = utils.group_by(array=det_level_tracks_source.data(), by=identifiers)
    part_level_tracks = utils.group_by(array=part_level_tracks_source.data(), by=identifiers)
    # There is one entry per event, so we don't need to do any group by steps.
    event_properties = event_properties_source.data()

    # Event selection
    # We apply the event selection implicitly to the particles by requiring the identifiers
    # of the part level, det level, and event properties match.
    # NOTE: Since we're currently using this for conversion, it's better to have looser
    #       conditions so we can delete the original files. So we disable this for now.
    # event_properties = event_properties[
    #     (event_properties["is_ev_rej"] == 0)
    #     & (np.abs(event_properties["z_vtx_reco"]) < 10)
    # ]

    # Now, we're on to merging the particles and event level information. Remarkably, the part level,
    # det level, and event properties all have a unique set of identifiers. None of them are entirely
    # subsets of the others. This isn't particularly intuitive to me, but in any case, this seems to
    # match up with the way that it's handled in pyjetty.
    # If there future issues with merging, check out some additional thoughts and suggestions on
    # merging here: https://github.com/scikit-hep/awkward-1.0/discussions/633

    # First, grab the identifiers from each collection so we can match them up.
    # NOTE: For the tracks, everything is broadcasted to the shape of the particles, which is jagged,
    #       so we take the first instance for each event (since it will be the same for every particle
    #       in an event).
    det_level_tracks_identifiers = det_level_tracks[identifiers][:, 0]
    part_level_tracks_identifiers = part_level_tracks[identifiers][:, 0]
    event_properties_identifiers = event_properties[identifiers]

    # Next, find the overlap for each collection with each other collection, storing the result in
    # a mask.  As noted above, no collection appears to be a subset of the other.
    # Once we have the mask, we immediately apply it.
    # NOTE: isin doesn't work for a standard 2D array because a 2D array in the second argument will
    #       be flattened by numpy.  However, it works as expected if it's a structured array (which
    #       is the default approach for Array conversion, so we get a bit lucky here).
    det_level_tracks_mask = np.isin(  # type: ignore
        np.asarray(det_level_tracks_identifiers),
        np.asarray(part_level_tracks_identifiers),
    ) & np.isin(  # type: ignore
        np.asarray(det_level_tracks_identifiers),
        np.asarray(event_properties_identifiers),
    )
    det_level_tracks = det_level_tracks[det_level_tracks_mask]
    part_level_tracks_mask = np.isin(  # type: ignore
        np.asarray(part_level_tracks_identifiers),
        np.asarray(det_level_tracks_identifiers),
    ) & np.isin(  # type: ignore
        np.asarray(part_level_tracks_identifiers),
        np.asarray(event_properties_identifiers),
    )
    part_level_tracks = part_level_tracks[part_level_tracks_mask]
    event_properties_mask = np.isin(  # type: ignore
        np.asarray(event_properties_identifiers),
        np.asarray(det_level_tracks_identifiers),
    ) & np.isin(  # type: ignore
        np.asarray(event_properties_identifiers),
        np.asarray(part_level_tracks_identifiers),
    )
    event_properties = event_properties[event_properties_mask]

    # Now, some rearranging the field names for uniformity.
    # Apparently, the array will simplify to associate the three fields together. I assumed that a zip
    # would be required, but apparently not.
    return ak.Array(
        {
            "det_level": ak.zip(
                dict(
                    zip(
                        list(_standardized_particle_names.values()),
                        ak.unzip(det_level_tracks[list(_standardized_particle_names.keys())]),
                    )
                )
            ),
            "part_level": ak.zip(
                dict(
                    zip(
                        list(_standardized_particle_names.values()),
                        ak.unzip(part_level_tracks[list(_standardized_particle_names.keys())]),
                    )
                )
            ),
            **dict(zip(ak.fields(event_properties), ak.unzip(event_properties))),
        },
    )


def hf_tree_to_awkward_data(
    filename: Path,
    collision_system: str,
) -> ak.Array:
    # TODO: Consolidate with the _MC function
    # Setup
    # First, we need the identifiers to group_by
    # According to James:
    # Both data and MC need run_number and ev_id.
    # Data additionally needs ev_id_ext
    identifiers = [
        "run_number",
        "ev_id",
    ]
    if collision_system in ["pp", "PbPb"]:
        identifiers += ["ev_id_ext"]
    # Particle columns and names.
    particle_level_columns = identifiers + [
        "ParticlePt",
        "ParticleEta",
        "ParticlePhi",
    ]
    # We want them to be stored in a standardized manner.
    _standardized_particle_names = {
        "ParticlePt": "pt",
        "ParticleEta": "eta",
        "ParticlePhi": "phi",
    }

    # Detector level
    det_level_tracks_source = sources.UprootSource(
        filename=filename,
        tree_name="PWGHF_TreeCreator/tree_Particle",
        columns=particle_level_columns,
    )
    # Event level properties
    event_properties_columns = ["z_vtx_reco", "is_ev_rej"]
    # Collision system customization
    if collision_system == "PbPb":
        event_properties_columns += ["centrality"]
        # For the future, perhaps can add:
        # - event plane angle (but doesn't seem to be in HF tree output :-( )
    # It seems that the pythia relevant properties like pt hard bin, etc, are
    # all empty so nothing special to be done there.
    event_properties_source = sources.UprootSource(
        filename=filename,
        tree_name="PWGHF_TreeCreator/tree_event_char",
        columns=identifiers + event_properties_columns,
    )

    # Convert the flat arrays into jagged arrays by grouping by the identifiers.
    # This allows us to work with the data as expected.
    det_level_tracks = utils.group_by(array=det_level_tracks_source.data(), by=identifiers)
    # There is one entry per event, so we don't need to do any group by steps.
    event_properties = event_properties_source.data()

    # Event selection
    # We apply the event selection implicitly to the particles by requiring the identifiers
    # of the part level, det level, and event properties match.
    # NOTE: Since we're currently using this for conversion, it's better to have looser
    #       conditions so we can delete the original files. So we disable this for now.
    # event_properties = event_properties[
    #     (event_properties["is_ev_rej"] == 0)
    #     & (np.abs(event_properties["z_vtx_reco"]) < 10)
    # ]

    # Now, we're on to merging the particles and event level information. Remarkably, the part level,
    # det level, and event properties all have a unique set of identifiers. None of them are entirely
    # subsets of the others. This isn't particularly intuitive to me, but in any case, this seems to
    # match up with the way that it's handled in pyjetty.
    # If there future issues with merging, check out some additional thoughts and suggestions on
    # merging here: https://github.com/scikit-hep/awkward-1.0/discussions/633

    # First, grab the identifiers from each collection so we can match them up.
    # NOTE: For the tracks, everything is broadcasted to the shape of the particles, which is jagged,
    #       so we take the first instance for each event (since it will be the same for every particle
    #       in an event).
    det_level_tracks_identifiers = det_level_tracks[identifiers][:, 0]
    event_properties_identifiers = event_properties[identifiers]

    # Next, find the overlap for each collection with each other collection, storing the result in
    # a mask.  As noted above, no collection appears to be a subset of the other.
    # Once we have the mask, we immediately apply it.
    # NOTE: isin doesn't work for a standard 2D array because a 2D array in the second argument will
    #       be flattened by numpy.  However, it works as expected if it's a structured array (which
    #       is the default approach for Array conversion, so we get a bit lucky here).
    det_level_tracks_mask = np.isin(  # type: ignore
        np.asarray(det_level_tracks_identifiers),
        np.asarray(event_properties_identifiers),
    )
    det_level_tracks = det_level_tracks[det_level_tracks_mask]
    event_properties_mask = np.isin(  # type: ignore
        np.asarray(event_properties_identifiers),
        np.asarray(det_level_tracks_identifiers),
    )
    event_properties = event_properties[event_properties_mask]

    # Now, some rearranging the field names for uniformity.
    # Apparently, the array will simplify to associate the three fields together. I assumed that a zip
    # would be required, but apparently not.
    return ak.Array(
        {
            "data": ak.zip(
                dict(
                    zip(
                        list(_standardized_particle_names.values()),
                        ak.unzip(det_level_tracks[list(_standardized_particle_names.keys())]),
                    )
                )
            ),
            **dict(zip(ak.fields(event_properties), ak.unzip(event_properties))),
        },
    )


def write_to_parquet(arrays: ak.Array, filename: Path, collision_system: str) -> bool:
    """Write the jagged HF tree arrays to parquet.

    In this form, they should be ready to analyze.
    """
    # Determine the types for improved compression when writing
    # Ideally, we would determine these dyanmically, but it's unclear how to do this at
    # the moment with awkward, so for now we specify them by hand...
    # float_types = [np.float32, np.float64]
    # float_columns = list(self.output_dataframe.select_dtypes(include=float_types).keys())
    # other_columns = list(self.output_dataframe.select_dtypes(exclude=float_types).keys())
    # Typing info
    # In [8]: arrays.type
    # Out[8]: 18681 * {"det_level": var * {"pt": float32, "eta": float32, "phi": float32}, "part_level": var * {"pt": float32, "eta": float32, "phi": float32}, "run_number": int32, "ev_id": int32, "z_vtx_reco": float32, "is_ev_rej": int32}

    # Columns to store as integers
    use_dictionary = [
        "run_number",
        "ev_id",
        "is_ev_rej",
    ]
    if collision_system in ["pp", "PbPb"]:
        use_dictionary += ["ev_id_ext"]

    ak.to_parquet(
        array=arrays,
        where=filename,
        compression="zstd",
        # Use for anything other than floats
        use_dictionary=use_dictionary,
        # Optimize for floats for the rest
        # Generally enabling seems to work better than specifying exactly the fields
        # because it's unclear how to specify nested fields here.
        use_byte_stream_split=True,
        # use_byte_stream_split=[
        #     "pt", "eta", "phi",
        #     #"det_level", "part_level",
        #     "z_vtx_reco",
        # ],
    )

    return True


if __name__ == "__main__":
    # arrays = hf_tree_to_awkward(filename=Path("/software/rehlers/dev/substructure/trains/pythia/568/AnalysisResults.20g4.001.root"))
    # for collision_system in ["pythia"]:
    for collision_system in ["pp", "pythia", "PbPb"]:
        print(f"Converting collision system {collision_system}")
        if collision_system == "pythia":
            arrays = hf_tree_to_awkward_MC(
                filename=Path(
                    f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults.root"
                ),
                collision_system=collision_system,
            )
        else:
            arrays = hf_tree_to_awkward_data(
                filename=Path(
                    f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults.root"
                ),
                collision_system=collision_system,
            )

        write_to_parquet(
            arrays=arrays,
            filename=Path(
                f"/software/rehlers/dev/mammoth/projects/framework/{collision_system}/AnalysisResults.parquet"
            ),
            collision_system=collision_system,
        )
