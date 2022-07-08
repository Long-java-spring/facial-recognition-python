#!/bin/sh

echo "starting flask server..."

export PYTHONPATH=./src
python server.py run --host=0.0.0.0
