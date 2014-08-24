% Written by Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% Learn a GMM model using EM.  This algorithm tries to balance the clusters:
% any cluster that gets too small will be re-initialized by splitting one of
% the largest clusters.  Covariances are constrained to diagonal matrices.
%
% This is a memory-efficient function; it will not make a copy of data
% or allocate any arrays of that size.
%
% data: matrix of datapoints, one point per column.
% nctrs: number of centers to be output.
% conf: configuration with fields:
%   'convergence': if the sum-of-squared changes in centroid locations
%                  is less than this value, then the algorithm returns.
%
% res: struct with fields
%   'ctrs': the means of the GMM mixture components.  
%           size(res.ctrs)==[nctrs,size(data,1)]
%   'vars': the diagonals of the covariance matrices for each mixture component.
%           size(res.vars)==size(res.ctrs)

function res=gmmlearn(data,nctrs,conf);
if(~exist('conf','var'))
  conf=struct();
end
try
  rp=randperm(size(data,2));
  ctrs=data(:,rp(1:nctrs))';
  vars=repmat(var(data(:,rp(1:min(size(data,2),10000)))'),nctrs,1);
  if(dsbool(conf,'kmeans'))
    vars(:)=1;
  end
  %datasq=data.^2;
  if(dsfield(conf,'convergence'))
    convergence=conf.convergence;
  else
    meannrm=mean(sqrt(sum(data(:,rp(1:min(size(data,2),10000))).^2,1)),2);
    convergence=meannrm*nctrs/1000;
  end
  upd=Inf;
  reinit=[];
  while(true)
    sdata=zeros(nctrs,size(data,1));
    sdatasq=zeros(nctrs,size(data,1));
    ndata=zeros(nctrs,1);
    invvars=1./vars;
    for(batchst=1:1000:size(data,2))
      batch=batchst:min(batchst+999,size(data,2));
      ctrctr=sum(ctrs.*ctrs.*invvars,2);
      %ctrpts=ctrs.*invvars*data;
      %ptspts=invvars*datasq;
      probs=bsxfun(@plus,-(bsxfun(@plus,ctrctr,-2*(ctrs.*invvars*data(:,batch))+(invvars*data(:,batch).^2))),sum(log(invvars),2))/2;
      if(isnan(sum(sum(probs))))
        keyboard;
      end
      probs=exp(bsxfun(@minus,probs,max(probs,[],1)));
      if(isnan(sum(sum(probs))))
        keyboard;
      end
      probs=bsxfun(@rdivide,probs,sum(probs,1));
      if(isnan(sum(sum(probs))))
        keyboard;
      end
      %if(dsbool(conf,'kmeans'))
      %  [~,assn]=max(probs,[],1);
      %  probs=full(sparse(assn,1:numel(assn),ones(1,numel(assn))));
      %  sprobs0=sum(probs,2);
      %else
      %  sprobs0=sum(probs,2);
      %  probs=bsxfun(@rdivide,probs,sum(probs,2)+eps);
      %end
      if(isnan(sum(sum(probs))))
        keyboard;
      end
      if(upd<convergence&&numel(reinit)==0)
        break;
      end
      newctrs=ctrs;
      sprobsall=zeros(size(newctrs,1));
      reinit=[];
      %prpobthresh=[];
      %for(chk=1:100:size(probs,1));
      %  chki=chk:min(chk+99,size(probs,1));
      %  probthresh(chki,1)=quantile(probs(chki,:),100/size(probs,2),2);
      %end
      %probthresh=min(probthresh,1e-5);
      %[row,col]=find(bsxfun(@gt,probs,probthresh));
      %indsall=distributeby(col,row);

      for(i=1:size(ctrs,1))
        %inds=indsall{i};%find(probs(i,:)>1e-5);
        inds=find(probs(i,:)>1e-5);
        %if(~dsbool(conf,'kmeans'))
        %  [~,inds2]=maxk2(probs(i,:),100);
        %  inds=union(inds,inds2);
        %end
        %numel(inds)
        ndata(i)=ndata(i)+sum(probs(i,inds));
        sdata(i,:)=sdata(i,:)+(data(:,batch(inds))*c(probs(i,inds)))';
        sdatasq(i,:)=sdatasq(i,:)+(data(:,batch(inds)).^2*c(probs(i,inds)))';
        %if(sprobs0(i)<size(data,1)/8)
        %  reinit(end+1)=i;
        %end
        %newctrs(i,:)=(data(:,batch(inds))*(probs(i,inds)'))'./sprobs;
        %vars(i,:)=bsxfun(@minus,data(:,inds),newctrs(i,:)').^2*(probs(i,inds)')./sprobs+1e-3;
        if(any(vars(i,:)<1e-15))
          keyboard
        end
        %if(mod(i,50)==0)
        %  disp(i);
        %end
      end
    end
    if(upd<convergence&&numel(reinit)==0)
      break;
    end
    sprobs0=ndata;
    newctrs=bsxfun(@rdivide,sdata,ndata);
    vars=max(1e-3,bsxfun(@rdivide,sdatasq,ndata)-bsxfun(@rdivide,sdata,ndata).^2);

    reinit=find(sprobs0<.25*size(data,2)/nctrs);
    if(upd/2<convergence)
      reinit=[];
    end
    if(numel(reinit)>0)
      disp(['reinit ' num2str(numel(reinit))]);
      [~,split]=maxk(sprobs0,numel(reinit));
      for(i=1:numel(split))
        %[~,inds]=max(probs(split(i),:));
        newctrs(reinit(i),:)=data(:,floor(rand*size(data,2)))'*.001+.999*newctrs(split(i),:);
      end
      vars(reinit,:)=vars(split,:);
      oldsplit=split;
      oldreinit=reinit;
    end
    upd=sum(sum((newctrs-ctrs).^2));
    if(isnan(upd))
      keyboard
    end
    disp(['update:' num2str(upd)]);
    histc(sprobs0,1:200:max(sprobs0)+200)'
    ctrs=newctrs;
  end
  disp('done');
  %margmu=zeros(1,size(margdata,2),nctrs);
  %margcovar=zeros(size(margdata,2),size(margdata,2),nctrs);
  %for(i=1:size(ctrs,1))
  %  inds=find(probs(i,:)>1e-5);
  %  [margmu(:,:,i),margcovar(:,:,i)]=weightedgaussian(margdata(inds,:),probs(i,inds)');
  %end
  res.ctrs=ctrs;
  %if(~dsbool(conf,'kmeans'))
    res.vars=vars;
  %end
  %res.margmu=margmu;
  %res.margcovar=margcovar;
catch ex,dsprinterr;end
end
