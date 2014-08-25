#include <Eigen/Dense>
#include "mex.h"
#include "math.h"
#include <iostream>
#include <omp.h>
#include <vector>
#include "MinIndexedPQ.h"

using namespace std;

using Eigen::Matrix2d;
using Eigen::Vector2d;
using Eigen::VectorXd;
using Eigen::SelfAdjointEigenSolver;
using Eigen::Map;

// To cache the dot products that we need to compute between the
// query image features and the predictor image features.  Each
// cachecell is associated with one locaton in the query image
struct CacheCell{
  // stores the locations in the predictor image where the
  // cache is valid
  int xmin;
  int ymin;
  int xmax;
  int ymax;
  // The actual dot products.
  double* data;
  CacheCell():xmin(0),ymin(0),xmax(0),ymax(0),data(NULL){}
};

vector<vector<CacheCell> > cache;

//extern bool mxUnshareArray(const mxArray *pr, const bool noDeepCopy);

mxArray* getfield(mxArray* str, const string& fnam){
  int field_num = mxGetFieldNumber(str, fnam.c_str());
  return mxGetFieldByNumber(str, 0, field_num);
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  // Note that you should disable the parallel loop
  // before you enable debugging, or you will get segfaults!
  const bool DEBUG=false;
  // TODO: The first thing this mex function does is make a copy of its
  // input, which is really stupid because the structure of corresp does
  // not change, and the old values don't need to be kept.  Supposedly
  // these lines will let you avoid the copy, but I haven't tested it.
  //mxUnshareArray(const_cast<mxarray*> prhs[1],false);
  //mxArray* corresp=const_cast<mxarray*>(prhs[1]);
  mxArray* corresp=mxDuplicateArray(prhs[1]);
  int maxupdates = (int) mxGetScalar(prhs[3]);
  const mxArray* hogim = prhs[0];
  const int* hogimdims=mxGetDimensions(hogim);
  double* hogdata = mxGetPr(hogim);
  int ndims=hogimdims[0];
  int hogrows=hogimdims[1];
  int hogcols=hogimdims[2];
  int npyrs=* mxGetDimensions(prhs[2]);
  int* correspidx=(int *) mxGetData(prhs[4]);
  bool* inferred=(bool*)mxGetData(prhs[5]);
  int* inferinds=(int *)mxGetData(prhs[6]);
  int ntoinfer=*mxGetDimensions(prhs[6]);
  double* confidences = mxGetPr(prhs[7]);
  double lambda = mxGetScalar(prhs[8]);
  bool deleteCache = mxGetScalar(prhs[9])!=0;
  double* numneighbors = mxGetPr(prhs[10]);
  double lambdaprime = mxGetScalar(prhs[11]);
  if(DEBUG){mexPrintf("lam2 %f\n",lambda);mexEvalString("drawnow;");}
  if(DEBUG){mexPrintf("%d %d\n",ntoinfer,npyrs);}
  double** transf_out = new double*[npyrs];
  int ntransfs=0;
  // if deleteCache is 1, we're starting a new image, so clear out all
  // the cached dot products.
  if(deleteCache){
    for(int i = 0; i<cache.size(); ++i){
      for(int j=0; j<cache[i].size(); ++j){
        delete cache[i][j].data;
      }
    }
    cache.clear();
  }
  if(cache.size()==0){
    for(int i = 0; i<npyrs; ++i){
      cache.push_back(vector<CacheCell>(hogrows*hogcols));
    }
  }
  int cachehit=0;
  int cachemiss=0;
  string errmsg="";
  bool error=false;
  if(DEBUG){mexPrintf("start main loop over pyramids\n");}

  // Each pyramid's correspondence can be inferred in parallel.
  #pragma omp parallel for reduction(+:cachehit,cachemiss) shared(error,errmsg)
  for(int pyridx=0; pyridx<npyrs; ++pyridx){
    // These help us estimate when we need to return,
    // first by counting the number of times we've updated a 
    // mu and sigma pair, and second by seeing how long it's
    // been since something moved a lot.
    int nupdates=0;
    int nsincebigmove=0;

    // transf==alpha in the paper.  Figure out which cells need to have an
    // alpha value inferred before we can start updating mu's and sigmas.  
    // alpha's aren't kept around between calls to optimizecorresp, so
    // when we begin we need to infer an alpha for every cell in the 
    // condition region.
    // 
    // Likewise, cellheap associates with each cell an estimate of how
    // much that cell will change if it's updated.  We attempt to update
    // first any cell that seems like it will move a lot; this lets us
    // focus our computation on the parts of f that make the biggest
    // difference.
    vector<int> mustcomptransf;
    MinIndexedPQ cellheap(hogrows*hogcols);
    for(int ti = 0; ti<ntoinfer; ++ti){
      mustcomptransf.push_back(inferinds[ti]);
      cellheap.insert(inferinds[ti],-1000000);
    }
    vector<Matrix2d> transfs(hogrows*hogcols);
    if(DEBUG){cout<<"start main optimization loop\n";}
    // On each iteration of this loop, we update one mu and one sigma.
    while(!error){
      ++nupdates;
      ++nsincebigmove;

      // However, we may need to compute updates for many alphas, especially
      // at the beginning.
      for(int ti=0; ti<mustcomptransf.size(); ++ti){
        int infidx=mustcomptransf[ti];
        int xpos=infidx/hogrows;
        int ypos=infidx-xpos*hogrows;
        Matrix2d xsigma=Matrix2d::Zero();
        Matrix2d ysigma=Matrix2d::Zero();
        Vector2d xb=Vector2d::Zero();
        Vector2d yb=Vector2d::Zero();
        // Iterate over the edges of the lattice that this \alpha participates
        // in.  We aggregate statistics for both vertical and horizontal edges
        // simultaneously.  xsigma/ysigma aggregate the quadratic term
        // of equation 15, and xb/yb aggregate the linear term.
        for(int edgex=-2; edgex<=2; ++edgex){
          for(int edgey=-2; edgey<=2; ++edgey){
            if(edgex>-2&&edgex+xpos-1>=0&&edgex+xpos<hogcols&&edgey+ypos>=0&&edgey+ypos<hogrows &&
                  inferred[edgey+ypos+hogrows*(edgex+xpos-1)]&&inferred[edgey+ypos+hogrows*(edgex+xpos)]){
              mxArray* othcorresp=mxGetCell(corresp,edgey+ypos+(edgex+xpos)*hogrows);
              mxArray* othcorresp2=mxGetCell(corresp,edgey+ypos+(edgex+xpos-1)*hogrows);
              Map<Vector2d> othmu(mxGetPr(mxGetFieldByNumber(othcorresp, 0, 0))+2*pyridx);
              Map<Matrix2d> othcovar(mxGetPr(mxGetFieldByNumber(othcorresp, 0, 1))+4*pyridx);
              Map<Vector2d> othmu2(mxGetPr(mxGetFieldByNumber(othcorresp2, 0, 0))+2*pyridx);
              Map<Matrix2d> othcovar2(mxGetPr(mxGetFieldByNumber(othcorresp2, 0, 1))+4*pyridx);
              xsigma=xsigma+(othcovar.inverse()+othcovar2.inverse());
              xb=xb+(othcovar.inverse()+othcovar2.inverse())*(othmu-othmu2);
            }
            if(edgey>-2&&edgex+xpos>=0&&edgex+xpos<hogcols&&edgey+ypos-1>=0&&edgey+ypos<hogrows &&
                  inferred[edgey+ypos-1+hogrows*(edgex+xpos)]&&inferred[edgey+ypos+hogrows*(edgex+xpos)]){
              mxArray* othcorresp=mxGetCell(corresp,edgey+ypos+(edgex+xpos)*hogrows);
              mxArray* othcorresp2=mxGetCell(corresp,edgey+ypos-1+(edgex+xpos)*hogrows);
              Map<Vector2d> othmu(mxGetPr(mxGetFieldByNumber(othcorresp, 0, 0))+2*pyridx);
              Map<Matrix2d> othcovar(mxGetPr(mxGetFieldByNumber(othcorresp, 0, 1))+4*pyridx);
              Map<Vector2d> othmu2(mxGetPr(mxGetFieldByNumber(othcorresp2, 0, 0))+2*pyridx);
              Map<Matrix2d> othcovar2(mxGetPr(mxGetFieldByNumber(othcorresp2, 0, 1))+4*pyridx);
              ysigma=ysigma+(othcovar.inverse()+othcovar2.inverse());
              yb=yb+(othcovar.inverse()+othcovar2.inverse())*(othmu-othmu2);
            }
          }
        }
        transfs[ypos+xpos*hogrows].leftCols(1)=(xsigma).colPivHouseholderQr().solve(xb);
        transfs[ypos+xpos*hogrows].rightCols(1)=(ysigma).colPivHouseholderQr().solve(yb);
        if(DEBUG){cout<<"xsig\n"<<xsigma<<"\nxb\n"<<xb<<"\ntransfs\n"<<transfs[ypos+xpos*hogrows]<<"\n";}
      }
      mustcomptransf.clear();

      if(DEBUG){cout<<"start computing sigma's\n";}
      mxArray *mypyr=getfield(mxGetCell(prhs[2],pyridx),"features");
      // Get the linear index of the next mu/sigma pair to optimize, and store the estimated
      // change.
      double heapdist=cellheap.minKey();
      int infidx=cellheap.deleteMin();
      mxArray* mycorresp=mxGetCell(corresp,infidx);
      int edgex=1;
      int edgey=0;
      int xpos=infidx/hogrows;
      int ypos=infidx-xpos*hogrows;
      if(DEBUG){mexPrintf("get predictor feats\n");mexEvalString("drawnow;");}
      if(DEBUG){mexPrintf("%f\n",mxGetPr(mxGetFieldByNumber(mycorresp,0,2))[pyridx]);}
      mxArray* othhogarr=mxGetCell(mypyr,mxGetPr(mxGetFieldByNumber(mycorresp,0,2))[pyridx]-1);
      double* othhog=mxGetPr(othhogarr);
      const int* othhogdims=mxGetDimensions(othhogarr);
      int othrows=othhogdims[1];
      int othcols=othhogdims[2];
      // relsigma aggregates both Sigma in equation 21 and the outer product in equation 23.
      Matrix2d relsigma = Matrix2d::Zero();
      // relsigmainv aggregates inv(Sigma) in equation 21.
      Matrix2d relsigmainv = Matrix2d::Zero();
      if(DEBUG){mexPrintf("iterate over neighbors\n");mexEvalString("drawnow;");}
      Map<Vector2d> mymu(mxGetPr(mxGetFieldByNumber(mycorresp, 0, 0))+2*pyridx);
      Map<Matrix2d> mycovar(mxGetPr(mxGetFieldByNumber(mycorresp, 0, 1))+4*pyridx);
      // We first iterate over edges
      for(int edgeidx=0; edgeidx<4; ++edgeidx){
        if(DEBUG){mexPrintf("%d %d %d %d %d %d\n",edgex, edgey, xpos, ypos, hogrows, hogcols);}
        if(edgex+xpos>=0&&edgex+xpos<hogcols&&edgey+ypos>=0&&edgey+ypos<hogrows &&
          inferred[edgey+ypos+hogrows*(edgex+xpos)]){
          mxArray* othcorresp=mxGetCell(corresp,edgey+ypos+(edgex+xpos)*hogrows);
          Map<Vector2d> othmu(mxGetPr(mxGetFieldByNumber(othcorresp, 0, 0))+2*pyridx);
          Map<Matrix2d> othcovar(mxGetPr(mxGetFieldByNumber(othcorresp, 0, 1))+4*pyridx);
          Vector2d edge;
          edge << edgex, edgey;
          Matrix2d relsigmainv2=Matrix2d::Zero();
          Matrix2d relsigma2=Matrix2d::Zero();
          double n=0.0;
          // Then we iterate over the \alpha's that give us constraints over these edges
          for(int transfx=max(0,max(xpos,xpos+edgex)-2);transfx<=min(hogcols-1,min(xpos,xpos+edgex)+2); ++transfx){
            for(int transfy=max(0,max(ypos,ypos+edgey)-2);transfy<=min(hogrows-1,min(ypos,ypos+edgey)+2); ++transfy){
                if(DEBUG){cout << "tx " << transfx << " ty " <<transfy << "\n";}
              if(inferred[transfy+hogrows*(transfx)]){
                Vector2d mudiff=othmu-mymu-transfs[transfy+(transfx)*hogrows]*edge;
                if(DEBUG){cout << "othmu\n" << othmu << "\n" << mymu << "\n" << edge << "\ntransf\n"<<transfs[transfy+(transfx)*hogrows]<<"\n";}
                relsigma2=relsigma2+(othcovar+mudiff*mudiff.transpose())*lambdaprime;
                relsigmainv2=relsigmainv2+othcovar.inverse()*lambdaprime;
                n+=1.0;
              }
            }
          }
          // n is very nearly a constant and so in the paper I just rolled into lambdaprime.  However,
          // near the edge n will get smaller, and so dividing by n will give the warping
          // a little boost near the edge where there are fewer alpha's affecting each
          // edge.  
          relsigma=relsigma+relsigma2/n;
          relsigmainv=relsigmainv+relsigmainv2/n;
          Vector2d mudiff=othmu-mymu-edge;
          relsigma=relsigma+(othcovar+mudiff*mudiff.transpose())*lambda;
          relsigmainv=relsigmainv+othcovar.inverse()*lambda;
        }
        // We rotate the edge to get the next edge in the sequence.
        int tmp=edgey;
        edgey=edgex;
        edgex=-tmp;
      }

      // Next we need to aggregate the unary terms.  We actually do an e-step (equation 13) as we
      // go along; the values computed on the e-step are not stored.  
      //
      // First we need to compute a small region over which we should aggregate our statistics.
      // we use the same major/minor axis trick that we used in contextpredict to get
      // the window.
      SelfAdjointEigenSolver<Matrix2d> eigensolver(mycovar);
      Matrix2d V=eigensolver.eigenvectors().cwiseAbs();
      Vector2d D=eigensolver.eigenvalues();
      double mycovardet=mycovar.determinant();
      if(DEBUG){mexPrintf("computing eigenvalues 1\n");mexEvalString("drawnow;");}
      Matrix2d mycovarinv=mycovar.inverse();
      double dist=sqrt(fabs(1/(((1/(log(.0001)+log(mycovardet)/2))*(V.block(0,0,2,1).transpose()*mycovarinv*V.block(0,0,2,1)/2))(0))));
      double dist2=sqrt(fabs(1/(((1/(log(.0001)+log(mycovardet)/2))*(V.block(0,1,2,1).transpose()*mycovarinv*V.block(0,1,2,1)/2))(0))));
      if(DEBUG){mexPrintf("%f %f\n",dist,dist2);}
      Vector2d distvec = dist*V.block(0,0,2,1).cwiseMax(dist2*V.block(0,1,2,1));
      Map<VectorXd> mydata(hogdata+ndims*ypos+ndims*hogrows*xpos,ndims-1);
      double myconst = *(hogdata+ndims*ypos+ndims*hogrows*xpos+ndims-1);
      VectorXd mydatatransf=mydata;
      double sumprob=0;
      Matrix2d sumwt = Matrix2d::Zero();
      if(DEBUG){mexPrintf("5\n");mexEvalString("drawnow;");}
      //NOTE: window is in 1-indexed coordinates
      if(DEBUG){cout << distvec << "\n";}
      int xmin=floor(mymu(0)-min(distvec(0),30.0));
      int xmax=ceil(mymu(0)+min(distvec(0),30.0));
      int ymin=floor(mymu(1)-min(distvec(1),30.0));
      int ymax=ceil(mymu(1)+min(distvec(1),30.0));
      if(DEBUG){mexPrintf("5.1\n");mexEvalString("drawnow;");}

      // Now that we have the window, check to see if we have all the dot
      // products in that window cached.  We make sure that no more than 1/5
      // of the region that we need to compute is hanging outside of the
      // region stored in the cache on each side.
      CacheCell& mycache=cache[pyridx][hogrows*xpos+ypos];
      bool createcache=false;
      double* cachedata;
      if(DEBUG){mexPrintf("5.2\n");mexEvalString("drawnow;");}
      if(mycache.data!=NULL&&mycache.xmin<xmin+(xmax-xmin)/5&&mycache.ymin<ymin+(ymax-ymin)/5&&
         mycache.xmax>xmin+((xmax-xmin)*4+4)/5&&mycache.ymax>ymin+((ymax-ymin)*4+4)/5){
         ++cachehit;
      }else{
        ++cachemiss;
        createcache=true;
        mycache.xmin=xmin;
        mycache.ymin=ymin;
        mycache.xmax=xmax;
        mycache.ymax=ymax;
        if(mycache.data!=NULL){
          delete mycache.data;
        }
        mycache.data=new double[(ymax-ymin+1)*(xmax-xmin+1)];
      }
      if(DEBUG){mexPrintf("5.3\n");mexEvalString("drawnow;");}
      cachedata=mycache.data;
      if(DEBUG){mexPrintf("5.4\n");mexEvalString("drawnow;");}

      // Loop over the cache or the window, computing the inner products
      // and summing the total probability as required in equation 19.
      for(int windowx = max(xmin,mycache.xmin); windowx<=min(xmax,mycache.xmax); ++windowx){
        for(int windowy = max(ymin,mycache.ymin); windowy<=min(ymax,mycache.ymax); ++windowy){
          Vector2d window;
          window << windowx,windowy;
          Vector2d pt=window-mymu;
          double prob=1/(2*M_PI*sqrt(mycovardet))*exp((-pt.transpose()*mycovarinv*pt)(0)/2);
          if(prob!=prob){
            if(DEBUG){mexPrintf("%d\n",mycovardet);}
            errmsg="nan prob";
            if(DEBUG){mexErrMsgTxt(errmsg.c_str());}
            error=1;
            continue;
          }
          double tmpprob;
          if(createcache){
            int idx=ndims*(min(othrows-1,max(0,windowy-1))+min(othcols-1,max(0,windowx-1))*othrows);
            Map<VectorXd> othdata(othhog+idx,ndims-1);
            tmpprob = exp(-(mydatatransf.dot(othdata)+myconst)/2);
            tmpprob=tmpprob/(tmpprob+.01*(*(othhog+idx+ndims-1)));
            cachedata[(windowx-xmin)*(ymax-ymin+1)+windowy-ymin]=tmpprob;
            if(prob!=prob){
              if(DEBUG){mexPrintf("%f %f %f %f\n",tmpprob,myconst,mydatatransf.dot(othdata),*(othhog+idx+ndims-1));}
              if(DEBUG){cout<<"othdata\n"<<othdata<<"\nmydata\n"<<mydata<<"\n";}
              errmsg="nan prob v2";
              if(DEBUG){mexErrMsgTxt(errmsg.c_str());}
              error=1;
              continue;
            }
          }else{
            tmpprob=cachedata[(windowx-mycache.xmin)*(mycache.ymax-mycache.ymin+1)+windowy-mycache.ymin];
          }

          prob=prob*tmpprob;
          sumprob+=prob;
          sumwt=sumwt+prob*pt*(pt.transpose());
        }
      }
      if(DEBUG){mexPrintf("6\n");mexEvalString("drawnow;");}
      if(DEBUG){mexPrintf("sumwt\n");mexEvalString("drawnow;");}
      if(DEBUG){cout << sumwt << "\n" << sumprob << "\n";}
      if(DEBUG){mexPrintf("relsigma\n");mexEvalString("drawnow;");}
      if(DEBUG){cout << relsigma << "\n";}
      if(DEBUG){mexPrintf("sumwt\n");mexEvalString("drawnow;");}
      if(DEBUG){cout << sumwt << "\n";}
      if(DEBUG){mexPrintf("relsigmainv\n");mexEvalString("drawnow;");}
      if(DEBUG){cout << relsigmainv << "\n";}
      if(DEBUG){mexPrintf("mycovarinv\n");mexEvalString("drawnow;");}
      if(DEBUG){cout << mycovarinv << "\n";}
      double myconfidence = confidences[ypos+hogrows*(xpos)];
      if(DEBUG){mexPrintf("myconfidence\n");mexEvalString("drawnow;");}
      if(DEBUG){cout << myconfidence << "\n";}

      // Finally, put it all together
      Matrix2d nablasigma=(mycovarinv*(relsigma+sumwt*myconfidence)*mycovarinv-relsigmainv-mycovarinv*sumprob*myconfidence)/2;
      double covarstepsz=.1;
      if(nablasigma(0)!=nablasigma(0)){
        errmsg="nans in nablasigma";
        if(DEBUG){mexErrMsgTxt(errmsg.c_str());}
        error=1;
        continue;
      }
      // As is often the case with EM algorithms, we can end up with a covariance matrix shrinking to zero and
      // causing problems.  The right thing to do is to put a constraint on the algorithm that the determinant
      // of the inverse covariance must be less than 1, and correctly project onto the constraint set.  The wrong 
      // thing to do is to not take gradient steps when the determinant of the covariance matrix ends up less 
      // than 1.  However, out of sheer laziness I've done the latter.  In practice this happens very rarely.
      while(-nablasigma(0)*covarstepsz>mycovar(0)||-nablasigma(3)*covarstepsz>mycovar(3)||(mycovar+nablasigma*covarstepsz).determinant()<1){
        covarstepsz=covarstepsz*.5;
      }
      if(DEBUG){mexPrintf("nablasigma\n");mexEvalString("drawnow;");}
      if(DEBUG){cout << nablasigma << "\n";}
      Matrix2d newcovar=mycovar+nablasigma*covarstepsz;
      if(newcovar(0)>1e4||newcovar(4)>1e4){
        errmsg="exploding sigma";
        if(DEBUG){mexErrMsgTxt(errmsg.c_str());}
        error=1;
        continue;
      }
      Matrix2d newcovarinv=newcovar.inverse();
      Vector2d constr=Vector2d::Zero();
      Matrix2d syst=Matrix2d::Zero();

      if(DEBUG){mexPrintf("starting mu update\n");mexEvalString("drawnow;");}
      // Just like before, we start off by iterating over edges, this time
      // aggregating system of linear equations that's defined by taking the 
      // derivative of equation 17 (note that this system also needs to have 
      // some terms from  equation 16 added in, but 
      // we'll add those terms to the system later) and setting it to 0.
      //
      // We ultimately end up with an equation of the form syst*mu=constr,
      // where each term in the sums of equation 17 adds to both syst and constr.
      edgex=1;
      edgey=0;
      for(int edgeidx=0; edgeidx<4; ++edgeidx){
        if(DEBUG){mexPrintf("%d %d %d %d %d %d\n",edgex, edgey, xpos, ypos, hogrows, hogcols);mexEvalString("drawnow;");}
        if(edgex+xpos>=0&&edgex+xpos<hogcols&&edgey+ypos>=0&&edgey+ypos<hogrows &&
           inferred[edgey+ypos+hogrows*(edgex+xpos)]){
          mxArray* othcorresp=mxGetCell(corresp,edgey+ypos+(edgex+xpos)*hogrows);
          Map<Vector2d> othmu(mxGetPr(mxGetFieldByNumber(othcorresp, 0, 0))+2*pyridx);
          Map<Matrix2d> othcovar(mxGetPr(mxGetFieldByNumber(othcorresp, 0, 1))+4*pyridx);
          Vector2d edge;
          edge << edgex, edgey;
          if(DEBUG){mexPrintf("17\n");mexEvalString("drawnow;");}
          if(DEBUG){cout << othcovar << "\n";}
          if(DEBUG){cout << othmu << "\n";}
          Matrix2d othinvcovar=othcovar.inverse();
          Matrix2d syst2=Matrix2d::Zero();
          Vector2d constr2=Vector2d::Zero();
          double n=0.0;
          for(int transfx=max(0,max(xpos,xpos+edgex)-2);transfx<=min(hogcols-1,min(xpos,xpos+edgex)+2); ++transfx){
            for(int transfy=max(0,max(ypos,ypos+edgey)-2);transfy<=min(hogrows-1,min(ypos,ypos+edgey)+2); ++transfy){
              if(inferred[transfy+hogrows*(transfx)]){
                Vector2d mudiff=othmu-transfs[transfy+(transfx)*hogrows]*edge;
                constr2=constr2+(othinvcovar+newcovarinv)*(mudiff)*lambdaprime;
                syst2=syst2+(othinvcovar+newcovarinv)*lambdaprime;
                n+=1.0;
              }
            }
          }
          // n is very nearly a constant and so in the paper I just rolled it into lambdaprime.  However,
          // near the edge n will get smaller, and so dividing by n will give the warping
          // a little boost near the edge where there are fewer alpha's affecting each
          // edge.  
          constr=constr+constr2/n;
          syst=syst+syst2/n;

          Vector2d mudiff=othmu-edge;
          constr=constr+(othinvcovar+newcovarinv)*(mudiff)*lambda;
          syst=syst+(othinvcovar+newcovarinv)*lambda;
        }
        int tmp=edgey;
        edgey=edgex;
        edgex=-tmp;
      }
      if(DEBUG){mexPrintf("11\n");mexEvalString("drawnow;");}
      if(DEBUG){cout << constr << "\n";}
      if(DEBUG){cout << syst << "\n";}

      double newcovardet=newcovar.determinant();
      Vector2d mn = Vector2d::Zero();
      sumprob=0;

      // Now aggregate the contribution from the system from equation 16.  Computing
      // all the stuff for the e-step (including the window, the cache, etc.) is the
      // same as above.
      xmin=floor(mymu(0)-min(distvec(0),30.0));
      xmax=ceil(mymu(0)+min(distvec(0),30.0));
      ymin=floor(mymu(1)-min(distvec(1),30.0));
      ymax=ceil(mymu(1)+min(distvec(1),30.0));
      mycache=cache[pyridx][hogrows*xpos+ypos];
      createcache=false;
      if(mycache.xmin<xmin+(xmax-xmin)/5&&mycache.ymin<ymin+(ymax-ymin)/5&&
         mycache.xmax>xmin+((xmax-xmin)*4+4)/5&&mycache.ymax>ymin+((ymax-ymin)*4+4)/5){
         ++cachehit;
      }else{
        ++cachemiss;
        createcache=true;
        mycache.xmin=xmin;
        mycache.ymin=ymin;
        mycache.xmax=xmax;
        mycache.ymax=ymax;
        if(mycache.data!=NULL){
          delete mycache.data;
        }
        mycache.data=new double[(ymax-ymin+1)*(xmax-xmin+1)];
      }
      cachedata=mycache.data;

      for(int windowx = max(xmin,mycache.xmin); windowx<=min(xmax,mycache.xmax); ++windowx){
        for(int windowy = max(ymin,mycache.ymin); windowy<=min(ymax,mycache.ymax); ++windowy){
          Vector2d window;
          window << windowx,windowy;
          Vector2d pt=window-mymu;
          double prob=1/(2*M_PI*sqrt(newcovardet))*exp((-pt.transpose()*newcovarinv*pt)(0)/2);
          double tmpprob;
          if(createcache){
            int idx=ndims*(min(othrows-1,max(0,windowy-1))+min(othcols-1,max(0,windowx-1))*othrows);
            Map<VectorXd> othdata(othhog+idx,ndims-1);
            tmpprob = exp(-(mydatatransf.dot(othdata)+myconst)/2);
            tmpprob=tmpprob/(tmpprob+.01*(*(othhog+idx+ndims-1)));
            cachedata[(windowx-xmin)*(ymax-ymin+1)+windowy-ymin]=tmpprob;
            if(prob!=prob){
              if(DEBUG){mexPrintf("%f %f %f %f\n",tmpprob,myconst,mydatatransf.dot(othdata),*(othhog+idx+ndims-1));}
              if(DEBUG){cout<<"othdata\n"<<othdata<<"\nmydata\n"<<mydata<<"\n";}
              errmsg="nan prob v3";
              if(DEBUG){mexErrMsgTxt(errmsg.c_str());}
              error=1;
              continue;
            }
          }else{
            tmpprob=cachedata[(windowx-mycache.xmin)*(mycache.ymax-mycache.ymin+1)+windowy-mycache.ymin];
          }
          prob=prob*tmpprob;
          sumprob+=prob;
          mn=mn+prob*pt;
        }
      }
      if(DEBUG){cout << mn << "\n" << mymu << "\n";}
      if(DEBUG){cout << sumprob << "\n";}
      constr=constr+newcovarinv*(mn+mymu*sumprob)*myconfidence;
      syst=syst+newcovarinv*sumprob*myconfidence;
      // Now put it all together.
      Vector2d newmu=syst.inverse()*constr;
      if(DEBUG){mexPrintf("Computed newmu, saving\n");mexEvalString("drawnow;");}
      if(DEBUG){mexPrintf("%f\n",infidx);mexEvalString("drawnow;");}

      mxArray* newcorresp=mxGetCell(corresp,infidx);
      Map<Vector2d> infmu(mxGetPr(mxGetFieldByNumber(newcorresp, 0, 0))+2*pyridx);
      Map<Matrix2d> infcovar(mxGetPr(mxGetFieldByNumber(newcorresp, 0, 1))+4*pyridx);
      if(DEBUG){cout << newmu << "\n";}
      if(DEBUG){cout << newcovar << "\n";}
      if(DEBUG){mexPrintf("%d\n",ntransfs);mexEvalString("drawnow;");}
      if(newmu(0)!=newmu(0)){
        errmsg="nans in newmu";
        if(DEBUG){mexErrMsgTxt(errmsg.c_str());}
        error=1;
        continue;
      }

      // At this point, actually updating the mu and sigma is finished. All that's left is to
      // put this cell back in the queue, and update all the other nodes that it's connected
      // to by an edge (since when this node moves, it will probably make its neighbors move).
      // In general, we aim to overestimate. so we don't have to be afraid of thinking we're
      // converged when we're not.
      double diff=(infmu-newmu).norm()*myconfidence;
      infmu=newmu;
      infcovar=newcovar;
      // We guess that, next time, the current node will move as much as it did this time.
      cellheap.insert(infidx,-diff);
      // Now, for each neighbor, we increment our estimate of how much it will move by the 
      // amount that this node moved, divided by the number of neighbors that node has.
      // This is in general a pretty bad estimate, but it's far better than just sweeping
      // over everything.
      for(int edge=0; edge<4; ++edge){
        if(DEBUG){mexPrintf("%d %d %d %d %d %d\n",edgex, edgey, xpos, ypos, hogrows, hogcols);mexEvalString("drawnow;");}
        if(edgex+xpos>=0&&edgex+xpos<hogcols&&edgey+ypos>=0&&edgey+ypos<hogrows &&
           inferred[edgey+ypos+hogrows*(edgex+xpos)]){
             cellheap.decreaseKey(edgey+ypos+hogrows*(edgex+xpos),cellheap.keyOf(edgey+ypos+hogrows*(edgex+xpos))-diff/numneighbors[edgey+ypos+hogrows*(edgex+xpos)]);
        }
        int tmp=edgey;
        edgey=edgex;
        edgex=-tmp;
      }
      // On the next round, re-infer some alphas.  Specifically, only re-infer the alpha
      // associated with the current node.  There's probably a better way to do this, but
      // in practice this seems to work well enough.
      mustcomptransf.push_back(infidx);

      // If we've run out of computation budget or we don't seem to be
      // moving very much anymore, return.  Copy the set of computed alpha's
      // from the stack to the heap so we can return them.
      if(diff>=.1){
        nsincebigmove=0;
      }
      if((nupdates>=maxupdates||nsincebigmove==100)&&heapdist>-100000){
        if(nlhs>=1){
          transf_out[pyridx]=new double[hogrows*hogcols*4];
          ++ntransfs;
          for(int i=0;i<hogrows*hogcols; ++i){
            Map<Matrix2d> transf_out_i(transf_out[pyridx]+4*i);
            transf_out_i=transfs[i];
          }
        }
        break;
      }
    }
  }
  if(error){
    mexErrMsgTxt(errmsg.c_str());
  }

  // Copy the alpha's from the heap to Matlab.
  plhs[0]=corresp;
  if(nlhs>=1){
    mxArray* tout=mxCreateCellMatrix(npyrs,1);
    for(int i=0; i<npyrs; ++i){
      int dims[4];
      dims[0]=2;
      dims[1]=2;
      dims[2]=hogrows;
      dims[3]=hogcols;
      mxArray* data=mxCreateNumericArray(4, dims, mxDOUBLE_CLASS, mxREAL);
      double* transf_out2=mxGetPr(data);
      for(int j=0; j<dims[2]*dims[3]*4; ++j){
        transf_out2[j]=transf_out[i][j];
      }
      mxSetCell(tout,i,data);
      delete transf_out[i];
    }
    plhs[1]=tout;
  }
  // Useful for debugging/performance analysis.
  mexPrintf("cache hit %d miss %d\n",cachehit,cachemiss);
  delete transf_out;
}
