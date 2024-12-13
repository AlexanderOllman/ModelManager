#!/bin/bash

echo "Pulling latest..."
git pull

echo "Building image..."
docker build -t fheonix/model-manager:0.0.3 .

echo "Pushing image..."
docker push fheonix/model-manager:0.0.3

echo "Done."


