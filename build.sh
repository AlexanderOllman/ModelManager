#!/bin/bash

echo "Pulling latest..."
git pull

echo "Building image..."
docker build -t fheonix/kanso:0.0.1 .

echo "Pushing image..."
docker push fheonix/kanso:0.0.1

echo "Done."


