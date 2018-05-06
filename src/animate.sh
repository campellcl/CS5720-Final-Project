#!/usr/bin/env bash
generateDemuxerFiles ()
{
    SOURCE_DIR=../results/PyTorchCNNVisualizations
    CLASSES_TO_ANIM=($(find ../results/PyTorchCNNVisualizations -maxdepth 1 -type d -printf '%P\n'))
    for class in "${CLASSES_TO_ANIM[@]}"; do
        if [ -e "demuxer_input_$class.txt" ]; then
            DEMUXER_FILE_NAME="demuxer_input_$class.txt"
            rm "$DEMUXER_FILE_NAME"
        fi
        if [ ! -e "demuxer_input_$class.txt" ]; then
             DEMUXER_FILE_NAME="demuxer_input_$class.txt"
             touch "$DEMUXER_FILE_NAME"
        fi
        getRelPath()
        {
            ANIM_OUT_PATH="$SOURCE_DIR/$class/anim"
            RELATIVE_PATH="$SOURCE_DIR/$class/plt"
            FILE_PATHS=($(find "$RELATIVE_PATH" -maxdepth 1 -type f))
#            echo "RELATIVE PATH '$RELATIVE_PATH'" >> "$DEMUXER_FILE_NAME"
#            echo "FILE PATHS '$FILE_PATHS'" >> "$DEMUXER_FILE_NAME"
        }
        getRelPath
        for file in "${FILE_PATHS[@]}"; do
            echo "file '$file'" >> "$DEMUXER_FILE_NAME"
            echo "duration 1" >> "$DEMUXER_FILE_NAME"
#            echo "FILE '$file'" >> "$DEMUXER_FILE_NAME"
        done
        echo "file '${FILE_PATHS[-1]}'" >> "$DEMUXER_FILE_NAME"
        echo "duration 1" >> "$DEMUXER_FILE_NAME"
        animateWithDemuxer ()
        {
            sh animate_gif.sh "$DEMUXER_FILE_NAME" "$ANIM_OUT_PATH/$class.mp4" "$ANIM_OUT_PATH/$class.gif"
        }
        animateWithDemuxer
        removeDemuxers ()
        {
            if [ -e "$DEMUXER_FILE_NAME" ]; then
                rm "$DEMUXER_FILE_NAME"
            fi
        }
        removeDemuxers
    done
}
generateDemuxerFiles
