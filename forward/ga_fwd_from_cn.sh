#!/bin/bash
clear
PYTHON=$(which python3)
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
#echo "Script path: $SCRIPTPATH"
now="$(date +'%d/%m/%Y %T')"

#================ Luanching one instance of "mqtt_fwd_influx.py" =======
# SERVICE="$SCRIPTPATH/spp_tools.py"
SERVICE1="$SCRIPTPATH/mqtt_fwd_influx.py GA_FORWARD_FROM_CN"
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

#================ Luanching another instance of "mqtt_fwd_influx.py" with different configurtion file=======
# SERVICE2="$SCRIPTPATH/mqtt_fwd_influx.py smartPP_fwd_conf.yaml"
# process2=$(pgrep -f "$SERVICE2")

# if [[ ! -z $process2 ]]
# then
#     echo "$SERVICE2 is running at $now"
# else
# #    echo "$SERVICE stopped at $now, restart!"
# #    echo "$SERVICE stopped at $now, restart!" >> $SCRIPTPATH/$LOG
#     cd $SCRIPTPATH
#     # nohup $PYTHON $SERVICE2 &
#     $PYTHON $SERVICE2
# fi