# Sats Receiver
[![PyPI](https://img.shields.io/pypi/v/sats-receiver?logo=python&logoColor=white)][pypi_proj]
[![PyPI - Downloads](https://img.shields.io/pypi/dm/sats-receiver?logo=python&logoColor=white)][pypi_proj]
[![PyPI - License](https://img.shields.io/pypi/l/sats-receiver?logo=open-source-initiative&logoColor=white)][license]  
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/baskiton/sats-receiver/tests.yml?label=tests&logo=github)][tests]
[![Codecov branch](https://img.shields.io/codecov/c/gh/baskiton/sats-receiver/dev?logo=codecov)][codecov]
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/baskiton/sats-receiver/pypi-upload.yml?label=upload&logo=github)][upload]

[pypi_proj]: https://pypi.org/project/sats-receiver/
[license]: https://github.com/baskiton/sats-receiver/blob/main/LICENSE
[tests]: https://github.com/baskiton/sats-receiver/actions/workflows/tests.yml
[codecov]: https://app.codecov.io/gh/baskiton/sats-receiver
[upload]: https://github.com/baskiton/sats-receiver/actions/workflows/pypi-upload.yml

Satellites data receiver based on GNURadio

<!-- TOC -->
* [About](#About)
* [Requirements](#Requirements)
* [Installation](#Installation)
  * [From source](#From-source)
  * [From PYPI](#From-PYPI)
* [Usage](#Usage)
* [Configure](#Configure)
  * [observer](#observer)
  * [tle](#tle)
  * [receivers](#receivers)
    * [sats](#sats)
      * [frequencies](#frequencies)
        * [modulations](#modulations)
        * [decoders](#decoders)
          * [gr-satellites](#gr-satellites)
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

![](doc/NOAA-15_2023-05-11_03-30-41,734229_map.jpg "NOAA-15")


### Requirements
The program has only been tested on Linux. Work on Windows is not guaranteed!
* Python>=3.10 (or lower, see below)
* GNURadio>=3.10 (or lower if gr-soapy installed); GUI-modules is not required
* librtlsdr (if you use RTL-SDR)

### Installation
I recommended to use miniconda. So, first of all,
[install it.](https://docs.conda.io/en/latest/miniconda.html#linux-installers)

#### From source
```commandline
cd sats-receiver
conda create -n sats-receiver-env
conda activate sats-receiver-env
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda env update -f environment.yml
pip install -r requirements.txt
```

#### From PYPI
```commandline
conda create -n sats-receiver-env python
conda activate sats-receiver-env
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda install gnuradio gnuradio-satellites
pip install sats-receiver
```

### Usage
First, activate conda environment:  
`conda activate sats-receiver-env`

`python -u -m sats_receiver [-h, --help] [--exec EXECUTOR --exec_config CONFIG] [--log LOG] [--sysu SYSU] config`  
* `config` Config file path. See [Configure](#Configure)
* `-h, --help` Help message
* `--exec EXECUTOR` Python script path containing specified executor named `Executor`
* `--exec_config CONFIG` Executor specific config file path
* `--log LOG` Logging level, INFO default
* `--sysu SYSU` System Usages debug info timeout in seconds, 1 hour default

For example, simple command line to launch program:  
`python -u -m sats_receiver /path/to/config.json`  
You can copy the `default.json` config file from the root of the repository to a
location of your choice

Program home directory is `~/sats_receiver`  
Logfile saved to program home directory (`~/sats_receiver/logs`)  
Tle files stored to program home directory (`~/sats_receiver/tle`)  

### Configure
The configuration file is in JSON format.  
You can copy the `default.json` file from the root of the repository to a
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

| Field           | Type    | Description                                         |
|:----------------|:--------|:----------------------------------------------------|
| url             | String  | URL to TLE file                                     |
| update_period   | Number  | TLE Update period, hours.                           |
| ignore_checksum | Boolean | _Optional._ Ignore TLE checksum. `false` by default |


#### receivers
Each receiver object contain:

| Field              | Type            | Description                                                                                                                                        |
|:-------------------|:----------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|
| name               | String          | Name of the Receiver                                                                                                                               |
| source             | String          | String value for gr-soapy driver key, e.g. `rtlsdr`, `lime`, `uhd`, `remote`                                                                       |
| tune               | Number          | Receiver tune frequency, Hz                                                                                                                        |
| samp_rate          | Number          | Receiver sample rate, Hz                                                                                                                           |
| output_directory   | String          | Directory to save received files. You also might specify `~` symbol to specify User home directory                                                 |
| sats               | Array of Object | List of Satellites configurations (see [sats](#sats))                                                                                              |
| decim_power        | Integer         | _Optional._ Power (for 2) sample rate decimation. `0` by default                                                                                   |
| enabled            | Boolean         | _Optional._ Enable or Disable this Receiver. `true` by default                                                                                     |
| serial             | String          | _Optional._ Serial number of the receiver. Empty by default                                                                                        |
| biast              | Boolean         | _Optional._ Bias-T enable/disable. `false` by default. **WARNING! Be careful when enabling this option! Use only if you know what it is and why!** |
| gain               | Boolean         | _Optional._ Receiver gain, dB. `0` by default                                                                                                      |
| freq_correction    | Number          | _Optional._ Receiver frequency correction, PPM. `0.0` by default                                                                                   |
| freq_correction_hz | Integer         | _Optional._ Receiver frequency correction, Hz. `0` by default                                                                                      |


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

| Field               | Type            | Description                                                                                                              |
|:--------------------|:----------------|:-------------------------------------------------------------------------------------------------------------------------|
| freq                | Number          | Basic signal frequency, Hz                                                                                               |
| bandwidth           | Number          | Received signal bandwidth, Hz                                                                                            |
| demode_out_sr       | Number          | _Optional._ Demodulator out samplerate. Equal to `bandwidth` by default                                                  |
| enabled             | Boolean         | _Optional._ Enable/Disable this frequency. `true` by default                                                             |
| subname             | String          | _Optional._ Subname added to result filename. Empty by default                                                           |
| freq_correction     | Boolean         | _Optional._ Correction for basic frequency, Hz. `0` by default                                                           |
| mode                | String          | _Optional._ Modulation option (see [modulations](#modulations)). `RAW` by default                                        |
| decode              | String          | _Optional._ Decoder option (see [decoders](#decoders)). `RAW` by default                                                 |
| iq_waterfall        | Object          | _Optional._ Write also IQ waterfall for bandwidth. `none` by default. See below for object info                          |
| channels            | Array of Number | _Required only for **FSK**, **GFSK**, **GMSK** mode._ Demodulation baudrates, bps. `[1200, 2400, 4800, 9600]` by default |
| deviation_factor    | Number          | _Required only for **FSK**, **GFSK**, **GMSK** mode._ Deviation frequency factor (baudrate / factor), `5` by default     |
| grs_file            | String          | _Optional. Only for **SATS** decoder._ See [gr-satellites](#gr-satellites) for details                                   |
| grs_name            | String          | _Optional. Only for **SATS** decoder._ See [gr-satellites](#gr-satellites) for details                                   |
| grs_norad           | Integer         | _Optional. Only for **SATS** decoder._ See [gr-satellites](#gr-satellites) for details                                   |
| grs_tlm_decode      | Boolean         | _Optional. Only for **SATS** decoder._ Save decoded telemetry. `true` by default                                         |
| qpsk_baudrate       | Number          | _Required only for **(O)QPSK** mode._ (O)QPSK Baudrate, bps                                                              |
| qpsk_excess_bw      | Number          | _Optional. Only for **(O)QPSK** mode._ (O)QPSK Excess bandwidth. `0.35` by default                                       |
| qpsk_ntaps          | Integer         | _Optional. Only for **(O)QPSK** mode._ (O)QPSK number of taps. `33` by default                                           |
| qpsk_costas_bw      | Number          | _Optional. Only for **(O)QPSK** mode._ (O)QPSK Costas bandwidth. `0.005` by default                                      |
| sstv_wsr            | Number          | _Optional. Only for **SSTV** decoder._ SSTV work samplerate. `16000` by default                                          |
| sstv_sync           | Number          | _Optional. Only for **SSTV** decoder._ SSTV syncing. `true` by default                                                   |
| sstv_live_exec      | Number          | _Optional. Only for **SSTV** decoder._ SSTV live executing. `false` by default                                           |
| ccc_frame_size      | Number          | _Optional. Only for **CCSDSCC** decoder._ Frame size, bytes. `892` by default                                            |
| ccc_pre_deint       | Boolean         | _Optional. Only for **CCSDSCC** decoder._ Pre-Deinterleaving. `false` by default                                         |
| ccc_diff            | Boolean         | _Optional. Only for **CCSDSCC** decoder._ Differential Decoding. `true` by default                                       |
| ccc_rs_dualbasis    | Boolean         | _Optional. Only for **CCSDSCC** decoder._ Reed-Solomon Dualbasis. `false` by default                                     |
| ccc_rs_interleaving | Number          | _Optional. Only for **CCSDSCC** decoder._ Reed-Solomon Interleaving. `4` by default                                      |
| ccc_derandomize     | Boolean         | _Optional. Only for **CCSDSCC** decoder._ Descrambling. `true` by default                                                |
| quad_gain           | Number          | _Optional. Only for **QUAD**, **SSTV_QUAD** modes._ Quadrature demodulation gain. `1.0` by default                       |
| raw_out_format      | String          | _Optional. Only for **RAW** decoder._ WAV output format. `WAV` by default                                                |
| raw_out_subformat   | String          | _Optional. Only for **RAW** decoder._ WAV output subformat. `FLOAT` by default                                           |
| proto_deframer      | String          | _Optional. Only for **PROTO** decoder._ Name of the gr-satellites deframer. See [proto](#proto) for detail.              |
| proto_options       | String          | _Optional. Only for **PROTO** decoder._ Deframer options. See [proto](#proto) for detail.                                |

* `iq_waterfall` Create waterfall. Mapping with options (might be empty):
  * `fft_size` FFT size (int) `4096` by default
  * `mode` Waterfall mode:
    * `MEAN` (default)
    * `MAX_HOLD`
    * `DECIMATION`

#### modulations
* `RAW`
* `AM`
* `FM`
* `WFM`
* `WFM_STEREO`
* `QUAD`
* `SSTV_QUAD`
* `QPSK`
* `OQPSK`
* `FSK`
* `GFSK`
* `GMSK`

#### decoders
* `RAW` Saved to 2-channel float32 WAV file with `bandwidth` sample rate. Other parameters:
  * `raw_out_format` WAV output format:
    * `NONE` do no write
    * `WAV` default
    * `WAV64`
  * `raw_out_subformat` WAV output subformat:
    * `FLOAT` default
    * `DOUBLE`
    * `PCM_16`
    * `PCM_24`
    * `PCM_32`
    * `PCM_U8`
* `CSOFT` Constellation Soft Decoder - 1-channel binary int8. Suitable for further processing, for example, in SatDump. _Only for constellation-mode._
* `CCSDSCC` CCSDS Conv Concat Decoder - CADU data. Suitable for further processing, for example, in SatDump. _Only for constellation-mode._
* `APT` Sats-Receiver APT binary file format. See [APT](sats_receiver/systems/README.md#APT)
* `SSTV` SSTV saved to PNG image with EXIF. Supported modes:
  * Robot (24, 36, 72)
  * Martin (M1, M2, M3, M4)
  * PD (50, 90, 120, 160, 180, 240, 290)
  * Scottie (S1, S2, S3, S4)
* `SATS` See [gr-satellites](#gr-satellites) for details
* `PROTO` Satellite deframer based decoder. KISS file on output. See [proto](#proto) for detail. _Only for *FSK mode._
* ~~`LRPT`~~ Not implemented yet

##### gr-satellites
See [gr-satellites Documentation][grs-doc]  
**IMPORTANT:** For this decoder need to leave the `modulation` on `RAW`  

This decoder need to specify one of the parameters for recognize satellite option:
* grs_file - Path to your own [SatYAML-file][grs-satyaml]
* grs_name - Satellite name (may different from [sats name](#sats))
* grs_norad - Satellite NORAD ID

[List of builtin supported satellites][grs-satlist]  
Additionally supported satellites can be found in the [satyaml](satyaml) directory of this repository

[grs-doc]: https://gr-satellites.readthedocs.io/en/latest/
[grs-satyaml]: https://gr-satellites.readthedocs.io/en/latest/satyaml.html
[grs-satlist]: https://gr-satellites.readthedocs.io/en/latest/supported_satellites.html

##### proto
**IMPORTANT:** For this decoder the `modulation` need to be set on `*FSK`!

Supported deframers and their options:
* `AALTO1`:
  * `syncword_threshold`: number of bit errors allowed in syncword (int), `4` by default
* `AAUSAT4`:
  * `syncword_threshold`: `8` by default
* `AISTECHSAT_2`:
  * `syncword_threshold`: `4` by default
* `AO40_FEC`:
  * `syncword_threshold`: `8` by default
  * `short_frames`: use short frames (used in SMOG-P) (bool), `false` by default
  * `crc`: use CRC-16 ARC (used in SMOG-P) (bool), `false` by default
* `AO40_UNCODED`:
  * `syncword_threshold`: `3` by default
* `ASTROCAST_FX25`:
  * `syncword_threshold`: `8` by default
  * `nrzi`: use NRZ-I instead of NRZ (bool), `true` by default
* `AX100`:
  * `mode`: mode to use ('RS' or 'ASM') (string) REQUIRED!
  * `scrambler`: scrambler to use, either `CCSDS` or `none` (only for ASM mode) (str), `CCSDS` by default
  * `syncword`: syncword to use (str), `10010011000010110101000111011110` by default
  * `syncword_threshold`: `4` by default
* `AX25`:
  * `g3ruh_scrambler`: use G3RUH descrambling (boolean). REQUIRED!
* `AX5043`
* `BINAR1`:
  * `syncword_threshold`: `0` by default
* `CCSDS_CONCATENATED`:
  * `frame_size`: frame size (not including parity check bytes) (int) `223` by default
  * `precoding`: either `none` or `differential` for differential precoding (str) `none` by default
  * `rs_en`: If Reed-Solomon should be enabled or not (bool) `true` by default
  * `rs_basis`: Reed-Solomon basis, either `conventional` or `dual` (str) `dual` by default
  * `rs_interleaving`: Reed-Solomon interleaving depth (int) `1` by default
  * `scrambler`: scrambler to use, either `CCSDS` or `none` (str) `CCSDS` by default
  * `convolutional`: convolutional code to use (str) `CCSDS` by default. One of the following:
    * `CCSDS`
    * `NASA-DSN`
    * `CCSDS uninverted`
    * `NASA-DSN uninverted`
  * `syncword_threshold`: `4` by default
* `CCSDS_RS`:
  * `frame_size`: frame size (not including parity check bytes) (int) `223` by default
  * `precoding`: either `none` or `differential` for differential precoding (str) `none` by default
  * `rs_en`: If Reed-Solomon should be enabled or not (bool) `true` by default
  * `rs_basis`: Reed-Solomon basis, either `conventional` or `dual` (str) `dual` by default
  * `rs_interleaving`: Reed-Solomon interleaving depth (int) `1` by default
  * `scrambler`: scrambler to use, either `CCSDS` or `none` (str) `CCSDS` by default
  * `syncword_threshold`: `4` by default
* `DIY1`
* `ENDUROSAT`:
  * `syncword_threshold`: `0` by default
* `ESEO`:
  * `syncword_threshold`: `0` by default
* `FOSSASAT`:
  * `syncword_threshold`: `0` by default
* `GEOSCAN`:
  * `syncword_threshold`: `4` by default
* `GRIZU263A`:
  * `syncword_threshold`: `8` by default
* `HADES`:
  * `syncword_threshold`: `0` by default
* `HSU_SAT1`
* `IDEASSAT`
* `K2SAT`:
  * `syncword_threshold`: `0` by default
* `LILACSAT_1`:
  * `syncword_threshold`: `4` by default
* `LUCKY7`:
  * `syncword_threshold`: `1` by default
* ~~`MOBITEX`~~:
  * ~~`nx`: use NX mode (bool) `false` by default~~
* `NGHAM`:
  * `decode_rs`: use Reed-Solomon decoding (bool) `false` by default
  * `syncword_threshold`: `4` by default
* `NUSAT`:
  * `syncword_threshold`: `0` by default
* `OPS_SAT`
* `REAKTOR_HELLO_WORLD`:
  * `syncword_threshold`: `4` by default
  * `syncword`: `reaktor hello world` or `light-1` (str) `reaktor hello world` by default
* `SANOSAT`:
  * `syncword_threshold`: `0` by default
* `SAT_3CAT_1`:
  * `syncword_threshold`: `4` by default
* `SMOGP_RA`:
  * `frame_size`: size of the frame before FEC (int) REQUIRED!
  * `variant`: variant of the protocol to use (`SMOG-P` (`0`), `SMOG-1` (`6`) or `MRC-100` (`4`)) (str) `SMOG-1` by default
  * `syncword_threshold`: `-1` by default. Use `variant` defaults when <0
* `SMOGP_SIGNALLING`:
  * `new_protocol`: enable new protocol used in SMOG-1 (bool) `false` by default
  * `syncword_threshold`: `8` by default
* `SNET`:
  * `buggy_crc`: use buggy CRC implementation of S-NET (bool) `true` by default
  * `syncword_threshold`: `4` by default
* `SPINO`:
  * `syncword_threshold`: `0` by default
* `SWIATOWID`:
  * `syncword_threshold`: `0` by default
* `TT64`:
  * `syncword_threshold`: `1` by default
* `U482C`:
  * `syncword_threshold`: `4` by default
* `UA01`
* `USP`:
  * `syncword_threshold`: `13` by default
* `YUSAT`


### Map Shapes
Map shapes config file `map_shapes.json` can be found at the root of this repository.
Shapefiles can be downloaded from [Natural Earth](https://www.naturalearthdata.com/downloads/)

| Field      | Type             | Description                                                                        |
|:-----------|:-----------------|:-----------------------------------------------------------------------------------|
| shapes     | Array of Array   | _Optional._ List of shapes data (see [shapes](#shapes))                            |
| shapes_dir | String           | _Optional. Only when `shapes` specified._ Path to directory contains shapes file   |
| points     | Object of Object | _Optional._ Additional points to draw on map (see [points](#points))               |
| line_width | Number           | _Optional._ Overlay lines width, pixels. `1` by default                            |


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
