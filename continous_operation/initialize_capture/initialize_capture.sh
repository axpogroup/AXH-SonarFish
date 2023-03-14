v4l2-ctl --set-edid=file=/home/fish-pi/code/continous_operation/initialize_capture/1080p25edid --fix-edid-checksums;
sleep 10s;
v4l2-ctl --set-dv-bt-timings query; v4l2-ctl -V; v4l2-ctl -v pixelformat=UYVY
