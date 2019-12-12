#!/bin/sh

set -e

ABS_PATH=$(pwd)

CTC_LIB_DIR=$ABS_PATH/external/CTCWordBeamSearch
HAND_DIR=$ABS_PATH/src/main/backend/data_extraction/handwriting_extract/src/

TF_SEACH_LIB_NAME=TFWordBeamSearch.so

[ ! -d $CTC_LIB_DIR ] && echo "Directory $CTC_LIB_DIR DOES NOT exist. Have you cloned the submodules?" && exit 1

cd $CTC_LIB_DIR/cpp/proj/
./buildTF.sh

cp $CTC_LIB_DIR/cpp/proj/$TF_SEACH_LIB_NAME $HAND_DIR