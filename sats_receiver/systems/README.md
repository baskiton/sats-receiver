## Transmit Systems

All examples from this readme can be found in [example directory](../../example) of this repository

#### APT

File format

| Field    | Description                                                                  |
|:---------|:-----------------------------------------------------------------------------|
| TLE      | 3 lines of TLE data. Each line terminated by `\n`. Max length is 210 symbols |
| lat      | Observer latitude, degrees, double                                           |
| lon      | Observer longitude, degrees, double                                          |
| end_time | End time UTC timestamp, double                                               |
| data     | Image data width 2080, float                                                 |

If you need create PNG from apt, execute this code:

```python
import numpy as np
import pathlib
import sys
from PIL import Image
from sats_receiver.systems.apt import Apt

aptf = pathlib.Path('/path/to/file.apt')
apt = Apt.from_apt(aptf)
if apt.process():
    sys.exit(1)

img = Image.fromarray((apt.data * 255).clip(0, 255).astype(np.uint8), 'L')
img.save(aptf.with_suffix('.png'), 'png')
```

If you need additionally add map overlay to image. execute next code:

```python
import json
import numpy as np
import pathlib
import sys
from PIL import Image
from sats_receiver.systems.apt import Apt
from sats_receiver.utils import MapShapes

aptf = pathlib.Path('/path/to/file.apt')
apt = Apt.from_apt(aptf)
if apt.process():
    sys.exit(1)

cfg = json.load(open('/path/to/map_shapes.json'))
# map_shapes.json can be found at the root of this repository.
# it can also be a dict with the same parameters. See README for detail

msh = MapShapes(cfg)
apt.create_maps_overlay(msh)

img_overlay = apt.map_overlay
img_overlay = Image.fromarray(img_overlay, 'RGBA')

img = Image.fromarray((apt.data * 255).clip(0, 255).astype(np.uint8), 'L').convert('RGB')
img.paste(img_overlay, (apt.IMAGE_A_START, 0), img_overlay)
img.paste(img_overlay, (apt.IMAGE_B_START, 0), img_overlay)

img.save(aptf.with_stem(aptf.stem + '_map').with_suffix('.png'), 'png')
```
