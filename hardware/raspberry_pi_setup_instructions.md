# Raspberry Pi setup

# Packages
sudo apt update; sudo apt upgrade; sudo apt autoremove; sudo reboot;

# SW dev.
sudo apt install git
sudo ssh-keygen
sudo cat home/fish-pi/.ssh/id_rsa.pub
# Copy key to account https://dev.azure.com/Axpo-AXP/
# Clone repo using command from https://dev.azure.com/Axpo-AXP/AXH-Secret-Module-Development/_git/AXH-SonarFish
# Some troubleshooting was needed here to get the ssh to work, in the end I created ~/.ssh/config with: 
# 
# Host github.com
#  User git
#  Hostname github.com
#  IdentityFile ~/.ssh/id_rsa

# and ran this command
git config core.sshCommand "ssh -i ~/.ssh/id_rsa"


# setup venv
sudo apt install python3-venv python3-pip
# create venv, navigate to folder
python3 -m venv venv
# activate: 
source venv/bin/activate 

# img viewer
pip install timg


# ---- OTHER ----

sudo apt install v4l-utils
sudo apt install ffmpeg

# System monitoring
sudo apt install htop
sudo apt install s-tui
# sudo apt install lm-sensor

# Modify inputrc files (system and local) to scroll up history
# sudo nano /etc/.inputrc 
# sudo nano ~/.inputrc 

# add these lines
# "\e[A": history-search-backward
# "\e[B": history-search-forward

# Aliases
alias gcm='git commit -m'
alias gs='git status'
alias gpl='git pull'
alias gps='git push'
# Add to ~/.bashrc
