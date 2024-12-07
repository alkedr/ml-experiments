#!/bin/bash

set -eux

cd "$(dirname "$0")"

if ! command -v python3.12
then
	sudo apt-get update
	sudo apt-get install -y software-properties-common
	sudo add-apt-repository -y ppa:deadsnakes/ppa
	sudo apt-get install -y python3.12 python3.12-venv
fi

if [ ! -d venv ]
then
	python3.12 -m venv venv
fi

. venv/bin/activate

(
	mkdir -p tmp
	trap 'rm -rf tmp' EXIT
	TMPDIR="$(pwd)/tmp" nice pip install -r requirements.txt --no-cache-dir
)

"$@"
