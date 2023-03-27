# Raspberry Pi setup

# ---- SW DEV ----

# Packages
sudo apt update
sudo apt upgrade
sudo apt autoremove
sudo reboot

# SW dev. 
sudo apt install git
# Authentiation, reference: https://learn.microsoft.com/en-us/azure/devops/repos/git/use-ssh-keys-to-authenticate?view=azure-devops
ssh-keygen
cat home/fish-pi/.ssh/id_rsa.pub
# Copy key to account https://dev.azure.com/Axpo-AXP/
# Clone repo using command from https://dev.azure.com/Axpo-AXP/AXH-Secret-Module-Development/_git/AXH-SonarFish

# if the key is not found run this command to direct git to the key
git config core.sshCommand "ssh -i ~/.ssh/id_rsa"

# setup venv
sudo apt install python3-venv python3-pip
# create venv, navigate to folder
python3 -m venv venv
# activate: 
source venv/bin/activate 
# then install the requirements of the repo and set the pre-commit hook if applicable
pip install -r requirements.txt
# reference https://mingle.axpo.com/display/HTD/Repo+Creation+and+Content+at+HTD-A 
pre-commit install --hook-type pre-push

# Have jobs run at reboot with crontab
crontab -e
# add the following lines
@reboot sleep 120 && XDG_RUNTIME_DIR="/run/user/1000" /home/fish-pi/code/venv/bin/python3.10 
/home/fish-pi/code/continous_operation/src/main_recording_handler.py
@reboot sleep 120 && XDG_RUNTIME_DIR="/run/user/1000" /home/fish-pi/code/venv/bin/python3.10 
/home/fish-pi/code/continous_operation/src/main_orchestrator.py
# weekly reboot on Sunday 0:00, not used since only thing it would catch is a broken dataplicity and it might mess with the mounting. If the internet fails it will not reach azure and trigger a reboot anyway.
# 0 0 * * 0 /sbin/shutdown -r now

# ---- SONAR FISH ----
sudo apt install ffmpeg
sudo apt install v4l-utils

# ---- OTHER ----

# Ubuntu version
Ubuntu desktop LTS 22.04 https://ubuntu.com/tutorials/how-to-install-ubuntu-desktop-on-raspberry-pi-4#1-overview

# System monitoring
sudo apt install htop
# For temperature
sudo apt install s-tui 
# Browsing
sudo apt install chromium-browser

# ArgonOne Fan controller
https://github.com/wimpysworld/argon1-ubuntu

# Modify inputrc files (system and local) to scroll up history
# sudo nano /etc/inputrc 
# sudo nano ~/inputrc 

# add / uncomment these lines
"\e[A": history-search-backward
"\e[B": history-search-forward

# Aliases # Add to ~/.bashrc
alias gcm='git commit -m'
alias control='/home/fish-pi/code/venv/bin/python3.10 /home/fish-pi/code/continous_operation/src/controller.py -command'

