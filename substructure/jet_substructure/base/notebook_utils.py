""" Functionality that is useful for notebooks (and perhaps elsewhere)

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import base64
import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import uproot
from pachyderm import binned_data


logger = logging.getLogger(__name__)

# These are very useful
all_grooming_methods = [
    "leading_kt",
    "leading_kt_z_cut_02",
    "leading_kt_z_cut_04",
    "dynamical_z",
    "dynamical_kt",
    "dynamical_time",
    "dynamical_core",
    "soft_drop_z_cut_02",
    "soft_drop_z_cut_04",
]
leading_kt_grooming_methods = all_grooming_methods[:3]
dynamical_grooming_methods = all_grooming_methods[3:7]
soft_drop_grooming_methods = all_grooming_methods[7:]


def load_histograms(
    filename: str, collision_system: str, tag: str, base_path: Path, verbose: bool = False
) -> Dict[str, binned_data.BinnedData]:
    """Load histograms stored in a file.

    Args:
        filename: Name of the file to open.
        collision_system: Name of the collision system.
        tag: Tag under which the file is stored. For example, "RDF".
        base_path: Base directory under which the file is stored. Usually the output path.
        verbose: Be extra verbose. Default: False.
    Returns:
        Dict containing histograms converted to `binned_data.BinnedData`. The key under which the hist
            is stored is the used as the dict key.
    """
    input_filename = base_path / collision_system / tag / filename
    hists = {}
    with uproot.open(input_filename) as f:
        for k in f.keys(cycle=False):
            if verbose:
                logger.debug(f"Retrieving hist {k}")
            hists[k] = binned_data.BinnedData.from_existing_data(f[k])

    return hists


def output_dir_f(output_dir: Path, identifier: str) -> Path:
    """Format an output_dir path with a given identifier.

    Also ensures that the directory exists.

    Args:
        output_dir: Output dir containing a format identifier, `{identifier}`.
        identifier: Identifier to include in the path. Usually, it's the collision system,
            but it doesn't have to be.
    Returns:
        Output path formatted with the identifier.
    """
    p = Path(str(output_dir).format(identifier=identifier))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _image_to_base64(filename: Path) -> str:
    """Convert an image at the give path to base64.

    By converting, we can embed it into an img tag, which means that we can move a file around without any issues.

    Args:
        filename: Path to the file.
    Returns:
        Base64 encoded string (decoded as utf-8, so we can treat it as a normal str).
    """
    return base64.b64encode(open(filename, "rb").read()).decode("utf-8")


def display_images(
    rows: Sequence[Sequence[str]], fig_output_dir: Path, embed_with_base64: bool = False, render_display: bool = True
) -> Optional[str]:
    """Display stored images in a layout using HTML + CSS.

    For each row, the width is determined by the number of images, such that they their widths
    will sum to the full width. This can make images rather small, so we include a scaling transform
    to make the images zoomable. Clicking once enlarges them, and then clicking again returns them
    to their original size.

    For details on the scaling transform, see: https://stackoverflow.com/a/56401601/12907985.

    The function is inspired by `ipyplot`.

    Args:
        rows: Lists of filenames of images to be displayed. Each entry in the list corresponds to one row.
        fig_output_dir: Directory where the figures are stored.
        embed_with_base64: Embed the image into the html by encoding it via base64. Default: False.
        render_display: If True, display the HTML immediately. Otherwise, it's up to the user. Default: True.
    Returns:
        The compiled HTML containing the image. If the images are rendered, then we skip returning the HTML
        to avoid dumping the str into the notebook output.
    """
    # First, we define the CSS necessary for zooming into the image.
    # This could be cleaned up and improved, but it's fine for now.
    full_html = """
    <style>
    .click-zoom input[type=checkbox] {
      display: none
    }

    .click-zoom img {
      %margin: 100px;
      transition: transform 0.25s ease;
      cursor: zoom-in
    }

    .click-zoom input[type=checkbox]:checked~img {
      transform: scale(var(--scale-factor));
      %transform: scale(2.0);
      cursor: zoom-out
    }
    </style>
    """
    for row in rows:
        # For convenience, handle single strings. We convert it to a length one list.
        if isinstance(row, str):
            row = [row]

        # Display images with equal width.
        # Need to express width as percentage for CSS.
        # Using math.floor and the small decrease of 0.05 to ensure that we don't accidentally push
        # the images onto the next line in case of margin, padding, etc.
        width = math.floor((1 / len(row) * 100) - 0.05)

        # For smaller images, we need a larger scale factor.
        # By using the number of images, the scale factor makes each image approximately 100% of the width.
        scale_factor = len(row)

        for i, filename in enumerate(row):
            # Setup
            image_path = fig_output_dir / f"{filename}.png"
            # If there's only one image, there's no point in scaling. So we remove the class and the checkbox.
            zoom_class = 'class="click-zoom"'
            if len(row) == 1:
                zoom_class = ""

            # Determine transform location
            # We always use the top as origin because when the image expands down below the nominal bottom of the view,
            # scrollbars are added automatically. But if we go above the top, they aren't. This way, we can always have
            # a way to see the entire image (even if scrolling isn't super convenient).
            origin = "top"
            # If we're on the left, then we put the origin on the left. We use right for images on the right.
            # We also want the center to display with center. Note that there are different measures for even vs odd.
            if len(row) % 2 == 0:
                measure = len(row)
                # Even
                origin += " left" if i < (measure / 2) else " right"
            else:
                # Odd
                measure = len(row) - 1
                if i == (measure / 2):
                    origin += " center"
                else:
                    origin += " left" if i < (measure / 2) else " right"

            # Finally, specify the actual image source (and encode if necessary).
            image_src = str(image_path)
            if embed_with_base64:
                image_src = f"data:image/png;base64,{_image_to_base64(image_path)}"

            # After all of the steup, we can finally compose the HTML.
            html = f"""<div {zoom_class} style="display: inline-block; width: {width}%; vertical-align: top; text-align: center;">
            <label>
            """
            # Only add the checkbox (which will then be hidden by the CSS) if we're actually adding the scaling.
            if zoom_class:
                html += '<input type="checkbox">'
            html += f"""<img style="--scale-factor: {scale_factor}; transform-origin: {origin}; margin: 1px; width: 100%; border: 2px solid #ddd;" src="{image_src}"/>"""
            html += "</label></div>\n"
            full_html += html

    if render_display:
        from IPython.display import HTML, display

        display(HTML(full_html))
    else:
        return full_html

    return None


def display_images_ipywidgets(rows: Sequence[Sequence[str]], fig_output_dir: Path, render_display: bool = True) -> Any:
    """Display images with ipywidgets (backup method).

    This doesn't work nearly as well as the direct HTML based approach, but it's kept here for posterity.
    For one, it doesn't seem to render nicely when exporting notebooks. Plus, jupyter lab doesn't play
    entirely nicely with ipywidgets, even if it's supposed to support it. It's now also less sophisticated
    than the other method.

    Args:
        rows: Lists of filenames of images to be displayed. Each entry in the list corresponds to one row.
        fig_output_dir: Directory where the figures are stored.
        render_display: If True, display the HTML immediately. Otherwise, it's up to the user. Default: True.
    Returns:
        The compiled HTML containing the image.
    """
    # Delay the import so we don't have to rely on a package that's only used as a backup method.
    from ipywidgets import HBox, Image, VBox

    layout = []
    for row in rows:
        # For convenience, handle single strings
        if isinstance(row, str):
            row = [row]

        _images = [Image(value=open(fig_output_dir / f"{filename}.png", "rb").read(), format="png") for filename in row]
        layout.append(HBox(_images))

    ret_value = VBox(layout)
    if render_display:
        from IPython.display import display

        display(ret_value)

    return ret_value
