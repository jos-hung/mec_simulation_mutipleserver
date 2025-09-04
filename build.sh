#!/bin/bash

docker build -f docker/Dockerfile -t mec_simulation:latest .

BASE_PORT=10000

for i in {1..1}; do
    container_name="mec_simulation_$i"
    port=$((BASE_PORT + i))  

    echo "Starting container $container_name on port $port ..."

    docker run -dit \
        --network=host \
        -v $(pwd)/val2017:/shared \
        -v $(pwd)/service:/service \
        --name $container_name \
        mec_simulation:latest \
        

    echo "Sending install request to container $i ..."
done
wait
for i in {1..1}; do
    container_name="mec_simulation_$i"
    port=$((BASE_PORT + i))
    echo "Starting uvicorn inside container $container_name on port $port ..."
    osascript -e "tell application \"Terminal\" to do script \"docker exec -it $container_name bash -c 'PYTHONPATH=src uvicorn handle_host_request:app --host 0.0.0.0 --port $port'\""
    # gnome-terminal -- bash -c "docker exec -it $container_name bash -c 'PYTHONPATH=src uvicorn handle_host_request:app --host 0.0.0.0 --port $port'; exec bash"
done
wait
for i in {1..1}; do
    port=$((BASE_PORT + i))
    echo "Sending install request to docker $i on port $port ..."
    .venv/bin/python3 host_send_request.py \
        --request install \
        --num 1 \
        --docker $i \
        --host 0.0.0.0 \
        --port $port
done
wait
echo "All install requests sent."