#!/usr/bin/env bash
DEMUXER=$1
OUTFILE_MP4=$2
OUTFILE_GIF=$3

createLogFile ()
{
if [ ! -e anim_logfile.txt ]; then
    touch anim_logfile.txt
fi
#if [ -e anim_logfile.txt ]; then
#    rm anim_logfile.txt
#    touch anim_logfile.txt
#fi
}
createLogFile
animateWithDemuxers ()
{
#ffmpeg -f concat -i demuxer_input.txt -pix_fmt yuv420p -filter_complex loop=1:6 output.mp4
ffmpeg -f concat -safe 0 -y -i $DEMUXER -pix_fmt yuv420p $OUTFILE_MP4 $OUTFILE_GIF 2>> anim_logfile.txt
ffmpeg -y -i $OUTFILE_MP4 -loop 0 $OUTFILE_GIF
}
animateWithDemuxers
# This works:
# ffmpeg -f concat -i demuxer_input.txt -pix_fmt yuv420p output.mp4

#ffmpeg -r 1 -f concat -i demuxer_input.txt -vsync vfr -pix_fmt yuv420p output.mp4 -r 30
#ffmpeg -framerate 1 -f concat -i demuxer_input.txt -vsync vfr -pix_fmt yuv420p output.mp4 -r 30

