echo "SPARK_WORKLOAD: $1"
if [ "$1" = "Leader" ]; then
    # start master server
	start-master.sh -p 7077

    # start history server
    start-history-server.sh
else
	# start worker node
	WORKER_PORT=${2:-8081}
	echo "$WORKER_PORT"
	
    start-worker.sh spark://spark-master:7077
fi
