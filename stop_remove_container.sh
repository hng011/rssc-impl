#!/bin/bash

docker stop $(docker ps -q) 
docker rm $(docker ps -aq)
docker container ps -a