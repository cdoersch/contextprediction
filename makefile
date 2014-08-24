EIGEN3INCLUDE = -I/usr/include/eigen3/

MEXEXT = .$(shell mexext)

main: hog/features.$(MEXEXT) optimizecorresp.$(MEXEXT)

hog/features.$(MEXEXT):hog/features.cc
	mex -outdir hog hog/features.cc 
  
optimizecorresp.$(MEXEXT): optimizecorresp.cc
	#mex -v -I/usr/include/eigen3/ gausscorrespinfmex.cc
	#mex -v -I/usr/include/eigen3/ gausscorrespbeliefmex.cc CXXFLAGS="\$$CXXFLAGS -fopenmp" LDFLAGS="\$$LDFLAGS -fopenmp"
	mex -v $(EIGEN3INCLUDE) optimizecorresp.cc CXXFLAGS="\$$CXXFLAGS -fopenmp" LDFLAGS="\$$LDFLAGS -fopenmp"
