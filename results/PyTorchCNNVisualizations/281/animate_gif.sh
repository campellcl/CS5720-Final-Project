#!/usr/bin/env bash

ffmpeg -f concat -i demuxer_input.txt -pix_fmt yuv420p output.mp4
#ffmpeg -r 1 -f concat -i demuxer_input.txt -vsync vfr -pix_fmt yuv420p output.mp4 -r 30
#ffmpeg -framerate 1 -f concat -i demuxer_input.txt -vsync vfr -pix_fmt yuv420p output.mp4 -r 30

