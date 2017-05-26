# record screen action with recordmydesktop / gtk-recordmydesktop (rmd)
recordmydesktop

# speed out rmd's output and crop the video
# -ss seeks to start time x
# -t stops reading input at time x
ffmpeg -ss 00:00 -t 04:00 -i ~/out-6.ogv -filter:v "setpts=0.0625*PTS" -b 12000k -s 1600x832 out-6-sped-up-setpts-0.0625-PTS-HTSA2.mp4
