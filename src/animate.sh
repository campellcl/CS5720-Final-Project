#!/usr/bin/env bash
init ()
{
    SOURCE_DIR=../results/PyTorchCNNVisualizations/
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
    done
}
init

# Write demuxer_input:
if [ -e "$DEMUXER_FILE_NAME" ]; then
#    echo > demuxer_input.txt
#    echo "line 1" > "$DEMUXER_FILE_NAME"
#    rm "$DEMUXER_FILE_NAME"
    let i=1

    for dir in "${CLASSES_TO_ANIM[@]}"; do
        getRelPath()
        {
            RELATIVE_PATH="$SOURCE_DIR$dir/plt"
            FILE_PATHS=($(find "$RELATIVE_PATH" -maxdepth 1 -type f))
            echo "RELATIVE PATH '$RELATIVE_PATH'" >> "$DEMUXER_FILE_NAME"
#            echo "FILE PATHS '$FILE_PATHS'" >> "$DEMUXER_FILE_NAME"
        }
        getRelPath
        for file in "${FILE_PATHS[@]}"; do
            if [[ -f $file ]]; then
                echo "file '$file'" >> "$DEMUXER_FILE_NAME"
            fi
        done
#        echo "file '$SOURCE_DIR$dir/plt/'" >> "$DEMUXER_FILE_NAME"
    done

    echo "line 2" >> "$DEMUXER_FILE_NAME"
    printf "class: %d\n" 281 >> "$DEMUXER_FILE_NAME"
fi
# sh animate_gif.sh C:\Users\ccamp\Documents\GitHub\CS5720-Final-Project\results\PyTorchCNNVisualizations\281 paramOutput.mp4 paramOutputGIF.gif
