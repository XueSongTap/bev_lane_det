Dependency: 
1. OpenCV
2. Eigen

``` shell
cmake ./
make
./test
```

It tests the latency by transforming 1000 images into Virtual Camera, twice. 

+ This program will load the sample image named a.jpg.
+ The first test is only for warmup.
+ The transformed jpg will be written out in the name of vcam.jpg.