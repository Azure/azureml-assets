if [ "$1" = "Leader" ]; then
    # format HDFS NameNode
    if [[ -f /microsoft/hdfs/name/current/VERSION ]]; then
        echo "hdfs aleady formatted"
    else
        hdfs namenode -format
    fi

    # start the NameNode
    hdfs namenode &
    yarn resourcemanager &

    # start history server
    start-history-server.sh
fi

# start datanode
hdfs datanode &

# start node manager
yarn nodemanager &