#!/usr/bin/env bash
sudo chmod 755 $(pwd)/s3
docker_compose_path=$(which docker-compose)
$docker_compose_path up
