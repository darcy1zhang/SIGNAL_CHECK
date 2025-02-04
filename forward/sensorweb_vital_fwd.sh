#!/bin/bash
clear
PYTHON=$(which python3)
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
#echo "Script path: $SCRIPTPATH"
now="$(date +'%d/%m/%Y %T')"

#================ Luanching one instance of "vital_fwd.py" =======
# SERVICE="$SCRIPTPATH/spp_tools.py"
SERVICE1="$SCRIPTPATH/vital_fwd.py sensorweb_vital_fwd_conf.yaml"
# LOG="log.txt"
process1=$(pgrep -f "$SERVICE1")
# process=${process[0]}
# echo "the process ID of $mac is $process"

if [[ ! -z $process1 ]]
then
    echo "$SERVICE1 is running at $now"
else
#    echo "$SERVICE stopped at $now, restart!"
#    echo "$SERVICE stopped at $now, restart!" >> $SCRIPTPATH/$LOG
    cd $SCRIPTPATH
    nohup $PYTHON $SERVICE1 &
    # $PYTHON $SERVICE1
fi
