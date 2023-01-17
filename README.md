# Sats Receiver
[![PyPI](https://img.shields.io/pypi/v/sats_receiver?logo=python&logoColor=white)][pypi_proj]
[![PyPI - Downloads](https://img.shields.io/pypi/dm/sats_receiver?logo=python&logoColor=white)][pypi_proj]
[![PyPI - License](https://img.shields.io/pypi/l/sats_receiver?logo=open-source-initiative&logoColor=white)](https://github.com/baskiton/sats-receiver/blob/main/LICENSE)
[![upload](https://img.shields.io/github/actions/workflow/status/baskiton/sats_receiver/pypi-upload.yml?label=upload&logo=github)](https://github.com/baskiton/sats-receiver/actions/workflows/pypi-upload.yml)

[pypi_proj]: https://pypi.org/project/sats_receiver/

Satellites data receiver based on GNURadio

<!-- TOC -->
* [About](#About)
* [Requirements](#Requirements)
* [Installation](#Installation)
* [Usage](#Usage)
* [Configure](#Configure)
  * [observer](#observer)
  * [tle](#tle)
  * [receivers](#receivers)
    * [sats](#sats)
      * [frequencies](#frequencies)
        * [modulations](#modulations)
        * [decoders](#decoders)
* [Map Shapes](#Map-Shapes)
  * [shapes](#shapes)
  * [points](#points)
<!-- TOC -->

### About
This program is written to automate the process of receiving signals from
various orbiting satellites on your SDR. The basis for digital signal
processing is GNU Radio - a free software development toolkit that provides
signal processing blocks to implement software-defined radios and
signal-processing systems. [[wikipedia](https://en.wikipedia.org/wiki/GNU_Radio)]  
For example, this program is perfect for receiving weather
satellites like NOAA (image below).  
If you have ideas or knowledge on how to improve this project, feel free to submit issues or pull requests.

![](doc/NOAA-19_2023-01-12_04-24-35,752471_map.jpg "NOAA-19")


### Requirements
The program has only been tested on Linux. Work on Windows is not guaranteed!
* Python>=3.10 (or lower, see below)
* GNURadio>=3.10 (or lower if gr-soapy installed); GUI-modules is not required
* librtlsdr (if you use RTL-SDR)

### Installation
* if you need a virtual environment, you need to create it with the `--system-site-packages` option:  
  `python3 -m venv --system-site-packages venv`  
  `source venv/bin/activate`  
* from source  
  `git clone https://github.com/baskiton/sats-receiver.git`  
  `cd sats-receiver`  
  `pip install -r requirements.txt`  
* from pip  
  `pip install sats_receiver`  

### Usage
`python3 -m sats_receiver [-h, --help] [--log LOG] [--sysu SYSU] config`  
`sats_receiver [-h, --help] [--log LOG] [--sysu SYSU] config`  
* `config` Config file path. See [Configure](#Configure)
* `-h, --help` Help message
* `--log LOG` Logging level, INFO default
* `--sysu SYSU` System Usages info timeout in seconds, 1 hour default

Program home directory is `~/sats_receiver`
Logfile saved to program home directory (~/sats_receiver/logs)
Tle files stored to program home directory (~/sats_receiver/tle)

### Configure
The configuration file is in JSON format.  
You can copy the default.json file from the root of the repository to a
location of your choice and edit it.

| Field     | Type            | Description                                                |
|:----------|:----------------|:-----------------------------------------------------------|
| observer  | Object          | Observer/receiver parameters (see [observer](#observer))   |
| tle       | Object          | TLE data parameters (see [tle](#tle))                      |
| receivers | Array of Object | List of receivers parameters (see [receivers](#receivers)) |


#### observer

| Field     | Type           | Description                                                                                                         |
|:----------|:---------------|:--------------------------------------------------------------------------------------------------------------------|
| latitude  | Number         | Receiver Latitude, degrees                                                                                          |
| longitude | Number         | Receiver Longitude, degrees                                                                                         |
| elevation | Number or null | Receiver Elevation, meters. `null` means that the height will be obtained from the weather information or set to 0  |
| weather   | Boolean        | Whether to receive weather information from the Internet. The weather will be taken from the service open-meteo.com |


#### tle

| Field         | Type   | Description               |
|:--------------|:-------|:--------------------------|
| url           | String | URL to TLE file           |
| update_period | Number | TLE Update period, hours. |


#### receivers
Each receiver object contain:

| Field            | Type            | Description                                                                                                                                                                        |
|:-----------------|:----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| name             | String          | Name of the Receiver                                                                                                                                                               |
| source           | String          | String value for gr-soapy driver key, e.g. `rtlsdr`, `lime`, `uhd`, `remote`                                                                                                       |
| tune             | Number          | Receiver tune frequency, Hz                                                                                                                                                        |
| samp_rate        | Number          | Receiver sample rate, Hz                                                                                                                                                           |
| output_directory | String          | Directory to save received files. You also might specify `~` symbol to specify User home directory                                                                                 |
| sats             | Array of Object | List of Satellites configurations (see [sats](#sats))                                                                                                                              |
| enabled          | Boolean         | _Optional._ Enable or Disable this Receiver. `true` by default                                                                                                                     |
| serial           | String          | _Optional._ Serial number of the receiver. Empty by default                                                                                                                        |
| biast            | Boolean         | _Optional._ Bias-T enable/disable (only for RTL-SDR at this time). `false` by default. **WARNING! Be careful when enabling this option! Use only if you know what it is and why!** |
| gain             | Boolean         | _Optional._ Receiver gain, dB. `0` by default                                                                                                                                      |


#### sats
Each satellite object contain:

| Field         | Type            | Description                                                                                                |
|:--------------|:----------------|:-----------------------------------------------------------------------------------------------------------|
| name          | String          | Name or NORAD number of the satellite. Note: name/norad-number must be contained in the above TLE file     |
| frequencies   | Array of Object | List of frequency configuration (see [frequencies](#frequencies))                                          |
| enabled       | Boolean         | _Optional._ Enable/Disable this frequency. `true` by default                                               |
| min_elevation | Number          | _Optional._ Elevation angle above the horizon, degrees. `0` by default. Negative number is equivalent to 0 |
| doppler       | Boolean         | _Optional._ Enable/Disable doppler correction. `true` by default                                           |


#### frequencies
Each frequency object contain:

| Field           | Type    | Description                                                                       |
|:----------------|:--------|:----------------------------------------------------------------------------------|
| freq            | Number  | Basic signal frequency, Hz                                                        |
| bandwidth       | Number  | Received signal bandwidth, Hz                                                     |
| enabled         | Boolean | _Optional._ Enable/Disable this frequency. `true` by default                      |
| freq_correction | Boolean | _Optional._ Correction for basic frequency, Hz. `0` by default                    |
| mode            | String  | _Optional._ Modulation option (see [modulations](#modulations)). `RAW` by default |
| decode          | String  | _Optional._ Decoder option (see [decoders](#decoders)). `RAW` by default          |
| qpsk_baudrate   | Number  | _Required only for **QPSK** mode._ QPSK Baudrate, bps                             |
| qpsk_excess_bw  | Number  | _Optional. Only for **QPSK** mode._ QPSK Excess bandwidth. `0.35` by default      |
| qpsk_ntaps      | Integer | _Optional. Only for **QPSK** mode._ QPSK number of taps. `33` by default          |
| qpsk_costas_bw  | Number  | _Optional. Only for **QPSK** mode._ QPSK Costas bandwidth. `0.005` by default     |


#### modulations
* `RAW`
* `AM`
* `FM`
* `WFM`
* `WFM_STEREO`
* `QUAD`
* `QPSK`

#### decoders
* `RAW` Saved to 2-channel float32 WAV file with `bandwidth` sample rate
* `RSTREAM` Raw Stream - binary int8. Suitable for further processing, for example, in SatDump
* `APT` sats_receiver APT binary file format. See [APT](sats_receiver/systems/README.md#APT)
* ~~`LRPT`~~ Not implemented yet


### Map Shapes
Map shapes config file `map_shapes.json` can be found at the root of this repository.
Shapefiles can be downloaded from [Natural Earth](https://www.naturalearthdata.com/downloads/)

| Field      | Type             | Description                                                          |
|:-----------|:-----------------|:---------------------------------------------------------------------|
| shapes_dir | String           | Path to directory contains shapes file                               |
| shapes     | Array of Array   | List of shapes data (see [shapes](#shapes))                          |
| line_width | Number           | _Optional._ Overlay lines width, pixels. `1` by default              |
| points     | Object of Object | _Optional._ Additional points to draw on map (see [points](#points)) |


#### shapes
Each shape contain:

| Offset | Field     | Type                       | Description                                                                                                        |
|:-------|:----------|:---------------------------|:-------------------------------------------------------------------------------------------------------------------|
| 0      | order     | Number                     | Num in order of drawing. The more, the later it will be drawn.                                                     |
| 1      | shapefile | String                     | Filename of shapefile in shapes dir. Can be separates file or ZIP archive                                          |
| 2      | color     | String or Array of Integer | Color. Can be string representing (`red` e.g.), web hex (`#abcdef` e.g.) or 3-4-Array 0-255 (`[0, 127, 255]` e.g.) |

#### points
Each point object has name.  
If name is `observer`, then lonlat field is filled with lonlat from apt-file.  
Each point object contain:

| Field  | Type                        | Description                                                                                                                    |
|:-------|:----------------------------|:-------------------------------------------------------------------------------------------------------------------------------|
| color  | String or Array of Integer  | Color. Can be string representing (`red` e.g.), web hex (`#abcdef` e.g.) or 3-4-Array 0-255 (`[0, 127, 255]` e.g.)             |
| type   | String                      | Type of marker view. Can be `+`, `o`                                                                                           |
| size   | Integer or Array of Integer | If `type` is `+` then Array with line width and line length, pixels. If `type` is `o` then Integer as radius of circle, pixels |
| lonlat | Array of Number             | _Optional. **Only for non-observer name.**_ 2-Array of point longitude and latitude, degrees                                   |
| order  | Number                      | _Optional._ Same as in `shapes`. Default to last                                                                               |
