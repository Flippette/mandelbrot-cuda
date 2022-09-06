#!/bin/bash

pushd res
rm -rf *.ptx
nvcc *.cu -ptx -o lib.ptx
popd
