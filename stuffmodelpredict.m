% Generate a prediction using the stuff model.
%
% Given a condition region and a set of cells to use as
% prediction regions, generate a set of predictions.  
%
% hogim_orig: HOG features for the query image.  
%
% have_preds: binary mask specifying condition region (size(have_preds)=size(hogim_orig(:,:,1)))
% 
% xall/yall: x- and y-coordinates for the separate prediction regions.
%
% stuffmodel: the stuff model (see comments in contextpredict.m)
%
% condmu: the conditional means, one per page (size(condmu)=[1 size(hogim_orig,3) numel(xall)]).
%
% condmu: the conditional covariances, one per page (size(condmu)=[size(hogim_orig,3) size(hogim_orig,3) numel(xall)]).
function [condmu,condcovar]=stuffmodelpredict(hogim_orig,have_preds,xall,yall,stuffmodel)
try
    % Turns out you can get a fairly large speedup from vectorizing this--hence it's
    % vectorized.  I'm sorry.
    dims=size(hogim_orig,3);
    border=(size(stuffmodel.valid,1)-1)/2;
    featused=padarray(have_preds,[border,border])~=0;
    hogim_padded=padarray(hogim_orig,[border,border,0]);
    nearby=c(gendiamond(size(stuffmodel.valid,1)));
    % For each separate prediction region, figure out what needs to be assigned
    % to which Gaussian.
    %
    for(idx=1:numel(xall))
      x=xall(idx);
      y=yall(idx);
      % First, figure out which condition region cells are nearby, and
      % extract a patch the same size as the stuffmodel's valid field.
      cells2=c(featused(y:y+2*border,x:x+2*border))&nearby;
      hogpatch2=hogim_padded(y:y+2*border,x:x+2*border,:);
      hogpatch2=reshape(hogpatch2,[],dims);
      have_preds_patch=find(repmat(c(cells2),size(hogim_padded,3),1));

      % Next, find a GMM where the nearby cells
      % in the condition region will fit inside the region defined by the GMM.
      % If there's more than one, pick one at random.
      havebreak=false;
      for(modelidx=randperm(size(stuffmodel.valid,3)))
        valid=stuffmodel.valid(:,:,modelidx);
        if(all(valid(cells2)))
          havebreak=true;
          break;
        end
      end
      % Extract the patches in the same shape as the chosen stuffmodel.valid page, 
      % and keep track of which cells/feature dimensions we're actually allowed to
      % use out of those extracted (i.e. which were in the condition region).
      patches{idx}=c(hogpatch2(valid,:));
      usable{idx}=repmat(cells2(valid),dims,1);
      modelidxs(idx,1)=modelidx;

      % Note it's a pretty trivial change here if you want to actually do the factorization
      % correctly: just say that we're allowed to make the prediction for the next
      % index of xall/yall based on the current value using the following line.  However, we intentionally
      % break the stuff model factorization in the same way that we break the thing
      % model factorization; i.e. we predict independently within a given batch.
      %
      % featused(y+border,x+border)=1;
    end
    condmu=zeros([size(stuffmodel.gmm{modelidx}.condmu(:,:,1)) numel(modelidxs)]);
    condcovar=zeros([size(stuffmodel.gmm{modelidx}.condcovar(:,:,1)) numel(modelidxs)]);
    % Group the predictions we need to make by the GMM that will make them.
    [patches,usable,modelidxs,inverse]=distributeby(patches(:),usable(:),modelidxs);
    % Finally, actually make the predictions.  We use the common factorization trick:
    % c/det(\Sigma)exp((x-mu)\Sigma^-1(x-mu)')=c/det(\Sigma)exp(x\Sigma^-1x'-2mu\Sigma^-1x'+mu\Sigma^-1mu')
    % Remember \Sigma is diagonal.  We can mask out the various dimensions of x and mu selectively for
    % each point that needs to get assigned (in an axis-aligned GMM, marginalization is just as simple
    % as masking dimensions), and therefore we can vectorize by computing each term in the exponential
    % simultaneously for all patches and all GMM centers. These are the values in stored in 
    % ptspts, ctrpts, and ctrctr, respectively.
    for(idx=1:numel(modelidxs))
      modelidx=modelidxs(idx);
      ctrs=stuffmodel.gmm{modelidx}.ctrs;
      vars=stuffmodel.gmm{modelidx}.vars+1e-3;
      data=cell2mat(patches{idx}');
      valid=cell2mat(usable{idx}');
      datasq=data.^2;
      invvars=1./vars;

      ctrctr=(ctrs.*ctrs.*invvars)*valid;
      ctrpts=(ctrs.*invvars)*(data.*valid);
      ptspts=invvars*(datasq.*valid);
      probs=(-(ctrctr-2*ctrpts+ptspts)+(log(invvars)*valid))/2;
      probs=exp(bsxfun(@minus,probs,max(probs,[],1)));
      probs=bsxfun(@rdivide,probs,sum(probs,1));
      for(pt=1:size(probs,2))
        [probsidx]=find(probs(:,pt)>1e-4);
        myprobs=probs(probsidx,pt);
        myprobs=myprobs./sum(myprobs);
        condmutmp=stuffmodel.gmm{modelidx}.condmu(:,:,probsidx);
        condcovartmp=bsxfun(@plus,stuffmodel.gmm{modelidx}.condcovar(:,:,probsidx),eye(size(stuffmodel.gmm{modelidx}.condcovar,1))*.001);
        [condmu(:,:,inverse{idx}(pt)),condcovar(:,:,inverse{idx}(pt))]=combinegaussians2(condmutmp,condcovartmp,myprobs);
      end
    end
catch ex,dsprinterr;end
end
