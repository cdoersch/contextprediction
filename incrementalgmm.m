% Written by Carl DOersch (cdoersch at cs dot cmu dot edu)
%
% Aggregate sufficient statistics (sum, sum of outer products, number of 
% datapoints) for the conditional distribution
% of each component of each GMM in stuffmodelgmm.  Extracts all patches from
% the images listed in imidxs, and for each separate GMM in stuffmodelgmm, it
% extract the proper set of cells associated with that GMM, before assigning the
% patch to the components of that GMM.  The center of that patch then contributes
% to the conditional distribution associated with each of the components it was
% assigned to, proportional to the soft assignment scores. 
%
% featsum, dotsum, and ntotal are returned as cell arrays, where each cell
% corresponds to one component of one of the GMM's.  In these cell arrays,
% we first have the sufficient statistics for the first GMM, followed by
% the second GMM, and so on.
%
function [featsum,dotsum,ntotal]=incrementalgmm(stuffmodelgmm,imidxs,conf)
if(~exist('conf','var'))
  conf=struct();
end
try
featsum=repmat(repmat({0},size(stuffmodelgmm.gmm{1}.ctrs,1),1),size(stuffmodelgmm.valid,3),1);
dotsum=featsum;
ntotal=dotsum;
sz=size(stuffmodelgmm.gmm{1}.ctrs,1);
global ds;
if(isfield(stuffmodelgmm,'validtext'))
  valsz=size(stuffmodelgmm.validtext(:,:,1));
else
  valsz=size(stuffmodelgmm.valid(:,:,1));
end
for(i=1:numel(imidxs))
  params=ds.conf.params;
  levelFactor = params.levelFactor;
  %data = pos;
  %pos = pos.annotation;
  %rand('seed',1000*pos);
  I = im2double(getimg(ds,imidxs(i)));%imread([ds.conf.gbz{ds.conf.currimset}.cutoutdir ds.imgs(pos).fullpath]));


  if(dsfield(params,'imageCanonicalSize'))
    [IS, scale] = convertToCanonicalSize(I, params.imageCanonicalSize);
  else
    IS=I;
  end
  %[rows, cols, unused] = size(IS);
  %IG = getGradientImage(IS);
  pyramid = constructFeaturePyramidForImg(I, params);
  if(numel(pyramid.features)==0)
    continue;
  end
  hogsz=size(pyramid.features{1},3)
  pcs=valsz;%round(ds.conf.params.patchCanonicalSize/ds.conf.params.sBins)-2;
  [features, levels, indexes] = unentanglePyramid(pyramid, pcs,struct('normalizefeats',false,'normbeforewhit',false,'whitening',false));
  if(isempty(features))
    continue;
  end
  for(comp=1:size(stuffmodelgmm.valid,3))
    tic
    if(isfield(stuffmodelgmm,'validtext'))
      tmpfeat=hog2texture(features,stuffmodelgmm.hogctrs,stuffmodelgmm.valid(:,:,comp),stuffmodelgmm.validtext(:,:,comp));
    elseif(dsbool(ds.conf.params,'bigscalectr'))
      tmpfeat=features(:,[repmat(c(stuffmodelgmm.valid(:,:,comp)),(size(features,2)-hogsz)/numel(stuffmodelgmm.valid(:,:,comp)),1); true(hogsz,1)]>0);
    else
      tmpfeat=features(:,repmat(c(stuffmodelgmm.valid(:,:,comp)),size(features,2)/numel(stuffmodelgmm.valid(:,:,comp)),1)>0);
    end
    ctr=(c(stuffmodelgmm.valid(:,:,comp))*0)>0;
    ctr((numel(ctr)+1)/2)=true;
    if(dsbool(ds.conf.params,'bigscalectr'))
      ctrfeat=features(:,repmat(ctr,(size(features,2)-hogsz)/numel(ctr),1));
    else
      ctrfeat=features(:,repmat(ctr,size(features,2)/numel(ctr),1));
    end
    if(~dsbool(conf,'gmm'))
      dist=allpairsdist(stuffmodelgmm.gmm{comp}.ctrs,tmpfeat);
      [~,idx]=min(dist,[],1);
      [ctrfeat,ctridx]=distributeby(ctrfeat,idx(:));

      for(j=1:numel(ctridx))
        
        featsum{(comp-1)*sz+ctridx(j)}=featsum{(comp-1)*sz+ctridx(j)}+sum(ctrfeat{j},1);
        ntotal{(comp-1)*sz+ctridx(j)}=ntotal{(comp-1)*sz+ctridx(j)}+size(ctrfeat{j},1);
        dotsum{(comp-1)*sz+ctridx(j)}=dotsum{(comp-1)*sz+ctridx(j)}+ctrfeat{j}'*ctrfeat{j};
      end
    else
      invvars=1./stuffmodelgmm.gmm{comp}.vars;
      ctrs=stuffmodelgmm.gmm{comp}.ctrs;
      ctrctr=sum(ctrs.*ctrs.*invvars,2);
      ctrpts=ctrs.*invvars*tmpfeat';
      ptspts=invvars*(tmpfeat'.^2);
      probs=bsxfun(@plus,-(bsxfun(@plus,ctrctr,-2*ctrpts+ptspts)),sum(log(invvars),2))/2;
      probs=exp(bsxfun(@minus,probs,max(probs,[],1)));
      probs=bsxfun(@rdivide,probs,sum(probs,1));
      [ctridx,inds]=find(probs>.001);
      [inds,ctridx]=distributeby(inds,ctridx);
      for(j=1:numel(ctridx))
        myctrfeat=ctrfeat(inds{j},:);
        featsum{(comp-1)*sz+ctridx(j)}=featsum{(comp-1)*sz+ctridx(j)}+probs(ctridx(j),inds{j})*myctrfeat;
        ntotal{(comp-1)*sz+ctridx(j)}=ntotal{(comp-1)*sz+ctridx(j)}+sum(probs(ctridx(j),inds{j}));
        dotsum{(comp-1)*sz+ctridx(j)}=dotsum{(comp-1)*sz+ctridx(j)}+myctrfeat'*bsxfun(@times,c(probs(ctridx(j),inds{j})),myctrfeat);
      end
    end
    toc
  end
  %end
end
%if(ntotal==0)
%  return
%end
catch ex,dsprinterr;end
end
