% Optimize the correspondence datastructure given in corresp.  hogim is the
% query image feature vector (precomputed in contextpredict); pyrs
% is the same as in contextpredict; nrounds is the maximum number of individual cell
% updates that may be performed; confidence is the probability that
% each cell is part of the thing (c from the paper), conf specifies
% lambda and lambdaprime.
%
% transf contains the computed alpha's (local affine transformations) for
% debugging purposes.
%
% Internally, this just precomputes some convenience variables and then passes
% the input to the optimizecorresp mex function.

function [corresp,transf]=optimizecorrespwrap(hogim,corresp,pyrs,nrounds,confidence,conf)
try
  global ds
  if(~exist('conf','var'))
    conf=struct();
  end
  if(~all(size(hogim(1,:,:))==[1 size(corresp)]))
    error('hogim and corresp should be same size');
  end
  correspidx=zeros([size(corresp) numel(pyrs)],'int32');
  for(i=1:size(corresp,2))
    for(j=1:size(corresp,1))
      if(~isempty(corresp{j,i}))
        if(size(corresp{j,i}.mu,2)~=size(corresp{j,i}.covar,2))
          error('corresp dims dont match');
        end
      end
    end
  end
  inferred=~cellfun(@isempty,corresp);
  if(any(isnan(confidence(inferred))))
    error('nans in confidence');
  end
  if(isfield(conf,'posforinf'))
    posforinf=inferred.*conf.posforinf;
  else
    posforinf=inferred;
  end
  if(~exist('confidence','var'))
    confidence=double(inferred);
  end
  if(~all(size(confidence(:,:,1))==size(inferred)))
    error('confidence must be same size as corresp');
  end
  if(isfield(conf,'lambda'))
    lambda=conf.lambda;
  else
    lambda=1;
  end
  if(isfield(conf,'lambdaprime'))
    lambdaprime=conf.lambdaprime;
  else
    lambdaprime=1;
  end
  clearcache=double(dsbool(conf,'clearcache'))
  numneighborspercell=conv2(double(inferred),[0 1 0; 1 0 1; 0 1 0],'same');

  [corresp,transf]=optimizecorresp(hogim,corresp,pyrs(:),int32(nrounds),correspidx,inferred,int32(find(posforinf(:))-1),double(confidence),lambda,clearcache,numneighborspercell,lambdaprime);
  for(i=1:numel(transf))
    transf{i}=permute(transf{i},[3 4 1 2]);
  end
  catch ex,dsprinterr;
  end
end
