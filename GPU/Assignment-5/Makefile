file = Assignment-5.cu
CC = nvcc

all:
	$(CC) -Iinlude -Iinclude/GL -Linclude/lib -o Demo $(file) -lglew32 -lfreeglut

clean:
	del Demo.exe Demo.exp Demo.lib