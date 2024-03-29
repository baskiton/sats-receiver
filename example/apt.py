import json
import numpy as np
import pathlib
import sys

from typing import Union, Mapping

from PIL import Image
from sats_receiver.systems.apt import Apt
from sats_receiver.utils import MapShapes


def apt_to_png(apt_filename: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Create image from APT file
    :param apt_filename: path to APT file
    :return: Result file path
    """
    apt_filename = pathlib.Path(apt_filename)
    ret_fn = apt_filename.with_suffix('.png')
    apt = Apt.from_apt(apt_filename)

    if apt.process():
        sys.exit(1)

    img = Image.fromarray((apt.data * 255).clip(0, 255).astype(np.uint8), 'L')
    img.save(ret_fn, 'png')

    return ret_fn


def apt_to_png_map(apt_filename: Union[str, pathlib.Path], config: Union[pathlib.Path, str, Mapping]) -> pathlib.Path:
    """
    Create Image with map overlay from APT file
    :param apt_filename: path to APT file
    :param config: Config file path or dict/json object
    :return: Result file path
    """
    if isinstance(config, (str, pathlib.Path)):
        config = json.load(open(config))

    apt_filename = pathlib.Path(apt_filename)
    ret_fn = apt_filename.with_stem(apt_filename.stem + '_map').with_suffix('.png')
    apt = Apt.from_apt(apt_filename)

    if apt.process():
        sys.exit(1)

    msh = MapShapes(config)
    apt.create_maps_overlay(msh)

    img_overlay = apt.map_overlay
    img_overlay = Image.fromarray(img_overlay, 'RGBA')

    img = Image.fromarray((apt.data * 255).clip(0, 255).astype(np.uint8), 'L').convert('RGB')
    img.paste(img_overlay, (apt.IMAGE_A_START, 0), img_overlay)
    img.paste(img_overlay, (apt.IMAGE_B_START, 0), img_overlay)

    img.save(ret_fn, 'png')

    return ret_fn


def apt_to_composites(apt_filename: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Create composite images based on WxToImg LUT's
    :param apt_filename: path to APT file
    :return: Result directory path
    """
    apt_filename = pathlib.Path(apt_filename)
    result_dir = apt_filename.parent / apt_filename.stem
    result_dir.mkdir(parents=True, exist_ok=True)
    ret_fn = result_dir / (apt_filename.stem + '.png')

    apt = Apt.from_apt(apt_filename)

    if apt.process():
        sys.exit(1)

    comps = 'HVC', 'HVCT', 'MSA', 'NO', 'HVCP', 'HVCTP', 'MSAP', 'SEA', 'THRM', 'BP'
    for c, img in apt.create_composites(*comps):
        img.save(ret_fn.with_stem(f'{ret_fn.stem}_{c}'), 'png')

    return result_dir


def apt_to_images(apt_filename: Union[str, pathlib.Path], config: Union[pathlib.Path, str, Mapping]) -> pathlib.Path:
    """
    Create Image with map overlay from APT file, also generate composite images
    :param apt_filename: path to APT file
    :param config: Config file path or dict/json object
    :return: Result directory path
    """
    if isinstance(config, (str, pathlib.Path)):
        config = json.load(open(config))

    apt_filename = pathlib.Path(apt_filename)
    result_dir = apt_filename.parent / apt_filename.stem
    result_dir.mkdir(parents=True, exist_ok=True)
    ret_fn = result_dir / (apt_filename.stem + '.png')

    apt = Apt.from_apt(apt_filename)

    if apt.process():
        sys.exit(1)

    img = Image.fromarray((apt.data * 255).clip(0, 255).astype(np.uint8), 'L')
    img.save(ret_fn, 'png')

    msh = MapShapes(config)
    apt.create_maps_overlay(msh)

    map_overlay = apt.map_overlay
    map_overlay = Image.fromarray(map_overlay, 'RGBA')

    img = img.convert('RGB')
    img.paste(map_overlay, (apt.IMAGE_A_START, 0), map_overlay)
    img.paste(map_overlay, (apt.IMAGE_B_START, 0), map_overlay)
    img.save(ret_fn.with_stem(ret_fn.stem + '_map'), 'png')

    comps = 'HVC', 'HVCT', 'MSA', 'NO', 'HVCP', 'HVCTP', 'MSAP', 'SEA', 'THRM', 'BP'
    for c, img in apt.create_composites(*comps):
        img.save(ret_fn.with_stem(f'{ret_fn.stem}_{c}'), 'png')

    return result_dir
