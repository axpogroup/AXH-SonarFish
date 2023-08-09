## Generic setup
### Ubuntu 22.04
[Ubuntu desktop LTS 22.04 Raspberry Pi installation instructions](https://ubuntu.com/tutorials/how-to-install-ubuntu-desktop-on-raspberry-pi-4#1-overview) 

#### Useful Ubuntu settings
* Modify inputrc files (system and local) to scroll up history in terminal
  * `sudo nano /etc/inputrc `
  * `sudo nano ~/inputrc`
  * add / uncomment these lines
    * `"\e[A": history-search-backward`
    * `"\e[B": history-search-forward`

### Install packages
* `sudo apt update`
* `sudo apt upgrade`
* `sudo apt install git`
* `sudo apt install python3-venv python3-pip`
* useful applications
  * System monitoring `sudo apt install htop`
  * Temperature `sudo apt install s-tui `
  * Browsing `sudo apt install chromium-browser`
* `sudo apt autoremove`
* Reboot

## Specific setup for Sonar fish detection
### Install packages for recording
* `sudo apt install ffmpeg`
* `sudo apt install v4l-utils`

### Make scripts autostart at reboot with crontab
* `crontab -e` do not run it with sudo, otherwise it will be at a root level
* add the following lines
  * `@reboot sleep 120 && XDG_RUNTIME_DIR="/run/user/1000" /home/fish-pi/code/venv/bin/python3.10/home/fish-pi/code/continous_operation/src/main_recording_handler.py`
  * `@reboot sleep 120 && XDG_RUNTIME_DIR="/run/user/1000" /home/fish-pi/code/venv/bin/python3.10/home/fish-pi/code/continous_operation/src/main_orchestrator.py`

### Add an alias for the controller.py script
* Aliases: add to the bottom of "~/.bashrc"
  * `alias control='/home/fish-pi/code/venv/bin/python3.10 /home/fish-pi/code/continous_operation/src/controller.py -command'`

### Configure watchdog
* Check if the watchdog exists: `ls -la /dev/watchdog*`
* Install the watchdog application `sudo apt-get install watchdog`
* Open the config file:  `sudo nano /etc/watchdog.conf`
* Uncomment the following lines: TODO, check this
  * `watchdog-device        = /dev/watchdog`
  * `watchdog-timeout = 10 `
  * `interval = 2`
  * `realtime  = yes` 
  * `priority = 1`
  * `file = /home/fish-pi/code/continous_operation/watchdog_food.csv `
  * `change = 1800` this will trigger the watchdog in case the file hasn't changed in 1800s = 30min
* Reboot
* Check if the watchdog is active: `service watchdog status`
* references: [Blog](https://blog.kmp.or.at/watchdog-for-raspberry-pi/), [Article](https://www.gieseke-buch.de/raspberrypi/eingebauten-hardware-watchdog-zur-ueberwachung-nutzen)

### Setup remote access with Dataplicity
* Follow the installation procedure outlined on https://docs.dataplicity.com/docs#installation-procedure
* Once the setup is done and you have opened the shell in your browser use `su fish-pi` to log in

### Automounting setup
* In order to automount the SSD, add the following line to /etc/fstab (sudo nano /etc/fstab):
* Note that this command will automount the first disk that is connected (sda1), since it is not clear how this is determined it is best to only connect one disk at a time
* Note that the output folder specified in continous_operation/settings/orchestrator_settings.yaml must correspond to the mounting point specified below:
`/dev/sda1 /media/fish-pi/PortableSSD1 exfat defaults,uid=1000 0 0`
* In the past there was issues with the permissions of the HDMI capture device after automounting, to solve them run these commands:
`sudo adduser username video`
`sudo usermod -a -G video username`
