cd res
del *.ptx
nvcc *.cu -ptx -o lib.ptx
cd ..
