# Sats Receiver
Satellites data receiver based on GNURadio

* [Requirements](#Requirements)
* [Installation](#Installation)
* [Usage](#Usage)
* [Configure](#Configure)

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

### Configure
