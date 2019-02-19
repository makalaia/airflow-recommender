#!/usr/bin/env bash
sudo chmod -R 777 $(pwd)/s3
docker_compose_path=$(which docker-compose)
$docker_compose_path up
