#!/bin/bash
# teegnn_build.sh

rm -rf /opt/phytium_optee/app/TEEGNN/host/bin

rm -rf /opt/phytium_optee/app/TEEGNN/host/*.o

rm -rf /opt/phytium_optee/app/TEEGNN/ta/out

cd /opt/phytium_optee/app

./build_app TEEGNN

cd /opt/phytium_optee

./build-all install

rm -rf /usr/lib/optee_armtz

cp -r /data/optee_armtz /usr/lib/optee_armtz