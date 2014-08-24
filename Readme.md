## Context as Supervisory Signal: Discovering Objects with Predictable Context

Code written by Carl Doersch, with advice from Abhinav Gupta and Alyosha Efros.
This is the authors' implementation of the algorithm described in 
[this work](http://graphics.cs.cmu.edu/projects/contextPrediction/) published in ECCV 2014.

This is unsupported research code, with no warranty or claim of suitability for 
any particular purpose.  However, you are encouraged to contact me (cdoersch at cs dot cmu dot edu)
if you have difficulties running it.  My goal is that this code should be useful.

Acknowledgements: Part of this code (Specifically the code for finding nearest neighbors patches) is 
based on work by (Saurabh Singh)[http://graphics.cs.cmu.edu/projects/discriminativePatches/], which 
was previously based on code by (Tomasz Malisiewicz)[http://www.cs.cmu.edu/~tmalisie/projects/iccv11/].

## Running the Code

0. **Prerequisites** 
  0. Linux (Windows and mac may work for single verifications, but it hasn't been tested; Windows will almost certainly not work with the distributed mining code).
  0. MATLAB (tested with 2012b)
  0. dswork (it is included here as a submodule; clone this code with 'git clone --recursive' and make sure the dswork directory is populated)
  0. eigen3 library and the associated header files for compiling (libeigen3-dev in ubuntu is sufficient)
0. **Compile MEX files**
  0. Modify the included makefile so that EIGEN3INCLUDE points to the path containing the 'Eigen' directory that contains the header files.
  0. Run make.
0. **To Run the Simple Demo**
  0. After compiling the MEX files, just run quick_demo.m  It will automatically download all data that's required for a demo.  The demo will show verification for a single patch depicting a car.  See quick_demo.m for more information on how to run the verification procedure on an arbitrary set of images, which will hopefully be useful for creating your own projects.  Note that running verification for a single patch cluster does not require dswork.Running the demo will download a 600MB file to the working directory.
0. **To Run the full mining pipeline**
  0. The full mining pipeline will require about 2000 CPU hours and about 20GB free disk space for the output directory.
  0. This code uses the dswork framework for parallelization.  Matlab can have trouble running many processes in parallel since the toolbox cache is not synchronized.  Hence I strongly recommend disabling it.  Do this either in settings->General->uncheck "Enable toolbox path cache" (and exit matlab to save your changes), or modify .matlab/VERSION/matlab.prf and add the line "GeneralUseToolboxCache=Bfalse" (or changing the line for GeneralUseToolboxCache if it exists). 
  0. Edit objectdiscovery_main.m to configure your distributed environment (which involves setting an output directory [which is shared between workers; all communication will happen through it], specifying the hostname where qsub can be run, telling the system how much RAM each machine has, and setting any additional options you want passed to qsub, as well as setting the path to the PASCAL dataset. Really not that bad).  Detailed instructions in objectdiscovery_main.m
  0. Run objectdiscovery_main.m

## Crash course on dswork

The mining algorithm is computationally expensive, and so we use the dswork framework for parallelism for objectdiscovery_main.m. The README in the dswork directory gives full documentation, but here's a tl;dr summary.

dswork has two main features. First, it establishes a mapping between some directory on the filesystem and the variable 'ds' in your workspace. Hence, you can call

dssetout('/tmp'); ds.mydirectory.myvariable=rand(100); dssave;

This causes the variable ds.myvariable to be saved to '/tmp/ds/mydirectory/myvariable.mat'. dswork supports filesystem command analogous to unix, including dsmv, dsdelete, dssymlink (though this implementation is incomplete), and dscd. To make the syntax as concise as possible, the format that things are saved in depends on the variable suffix--thus far, the suffixes img and html and txt have special meanings.

Second, dswork supports some basic distributed processing features, including multiple matlabs on one machine, and multiple matlabs on different machines. To use multiple machines, the directory where dswork saves its files needs to be shared among all machines you are using.

At a high level, dsmapredopen() sets up a pool of workers that are essentially stateless.  
Using dsrundistributed() or dsmapreduce() will assign work to each worker, allows the 
workers to load data from the shared storage, and tracks the variables that get 
written.  Note that these sessions can safely be interrupted with Ctrl-C. If the 
program terminates and one distributed job is rolled back, objectdiscovery_main.m is 
designed to safely pick up where it left off.

All of the experiments for this project were performed using Starcluster on EC2, which sets up an OGS cluster with data shared over nfs. See dsmapredopen for instructions on starting the distributed session.

## Understanding the Code

My coding style is developed around rapid prototyping, and is probably different from what you're used to. Most of the code is commented, but if you find something confusing, just ask me about it; I'll update the comments so it won't confuse others.  Here's a few patterns that I tend to use.

1) I generally use parallel arrays where other programmers would use arrays of structs or arrays of objects. This is the case because I often need quick access to all values of a single field. Matlab's struct arrays support this, but it is extremely inefficient. The distributeby/invertdistributeby have become a sort of swiss army knife for handling parallel arrays in my code. You should memorize what distributeby does.

2) To ease dealing with parallel arrays, the effstr... commands are designed to deal with a struct holding multiple parallel arrays (effstr means 'efficient replacement for matlab struct arrays'). The motivation is that I can add temporary data to an object and keep track of it alongside those objects, all with minimal modification of the code.

3) If I have a collection of n bounding boxes, they will be stored in an n-by-8 array with the following column order: [x1 y1 x2 y2 detection_score detector_id image_id flip]. x- and y- coordinates are in terms of pixels in the space returned by getimg. flipped detections have flip=1, but are still in terms of the coordinates of the un-flipped image. These are used so frequently in the code that this format is used without comment; you should memorize the order.
