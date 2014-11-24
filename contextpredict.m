% written by Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% The main verification algorithm for a single patch.  In summary, it requires only
% a query bounding box, a set of predictor bounding boxes, and a pre-trained GMM-based 
% 'stuff' model (which can be downloaded from our website), and it will use 
% the context around the bounding boxes to tell you whether the query bounding box
% contains the same thing as at least a few of the predictor images.  
%
% querybbox: A vector specifying a bounding box in the same form as the rest of this project,
%           which specifies the query patch.
%           For the purposes of this file, its format is [x1 y1 x2 y2 N/A N/A image_id flip], where
%           image_id is an index into ds.imgs{ds.conf.currimset}.  If flip=1, then it is assumed
%           that the image (along with the bounding box) needs to be flipped before 
%           running verification.  flip is optional; if conf.queryimage is set, then image_id
%           is ignored, the query image is read from conf.queryimage, and only the first
%           four columns are required.  Note that
%           the initial condition region removes once HOG cell from the border of the region
%           specified in querybbox, since the rest of the features are all that's used during
%           the detection (nearest neighbors) phase.
% 
% predictorbbox: A matrix where each column is in the same format as querybbox, specifying patches
%           in the predictor images that querybbox potentially corresponds to.  If conf.predictorimages
%           is set, then the 7'th column is ignored, and only the first four columns are required.
%
% both querybbox and predictorbbox are assumed to be approximately square.
%
% stuffmodel: a stuff model, with the following fields:
%                 - valid: an n-by-n-by-k tensor where each page is a binary matrix specifying
%                          which cells were used when training the GMM.  Given an n-by-n-by-d (where d is the HOG 
%                          dimension) patch p, you can extract the features that could be assigned
%                          to the i'th GMM component by p(find(repmat(stuffmodel.valid,[1,1,d]))). This also
%                          specifies the ordering of the features in the gmm field.
%                 - gmm: a cell array where each element is one dictionary.  Each cell has the fields:
%                     - ctrs: the mean of each GMM component.  Each row is one mean.
%                     - vars: each row is the diagonal of the covariance matrix for one GMM component.
%                     - condmu: each page is a one-by-d vector specifying the mean of the conditional
%                               distribution over a single HOG cell.
%                     - condcovar: each page is a d-by-d vector specifying the covariance of the conditional
%                                  distribution over a single HOG cell.
%                 - mu: the global mean of individual HOG cells across the full dataset.
%                 - covar: the global covariance of individual HOG cells across the full dataset.
%
% confidencemaps: a cell array parallel with the rows of predictorbbox.  Each cell may be empty, in which case it
%                 is not used.  Otherwise, it is interpreted as the probability that each HOG cell in the
%                 predictor image contains the 'thing' that should be found in the query image.  \pi^{i} in
%                 the paper.  Note that it is assumed to be the same size as one level of the HOG pyramid
%                 (floor(round(size(im)*2.^(-i/levels_per_octave))/sBins)-2) for integer i.  However, it
%                 can be any level and this function will automatically resize it to fit the size of the
%                 features extracted for the predictor images.
%
% conf: additional parameters.  Parameters not specified will be read from ds.conf.params.  These include
%       the arguments for constructFeaturePyramidForImg as well as optional 
%       parameters:
%
%       - maxpreds: maximum number of cells to predict (default Inf)
%       - maxcomptime: maximum amount of time in seconds before this function will return.
%       - lambda: lambda for the correspondence estimation algorithm
%       - lambdaprime: lambdaprime for the correspondence estimation algorithm
%       - queryimage: an RGB image to be used as the query image
%       - predictorimages: a cell array of images to be used as predictor images, 
%                          parallel with the rows of predictorbbox
%       - maxinf: maximum number of updates optimizecorresp can make per cell
%                 that needs to get updated. 
%       - disp: if set to 1, display a few figures that illustrate the algorithm's
%               progress.
% 
% All output arguments are given in terms on the HOG grid of size 
% (floor(round(size(im)*2.^(-i/levels_per_octave))/sBins)-2) for some integer i.  i is chosen
% so that the intial condition region is approx 8-by-8.  Note the -2 is due to the fact that HOG
% cells at the border of the image are not correctly normalized, and so constructFeaturePyramidForImg
% removes them.
%
% thingloglik: the log likelihood given by each cell for the foreground model where it predicted, 
%              and 0 elsewhere, without mimicry (\hat{p}^{T} in the paper).
%
% stuffloglik: the same as thingloglik, but for the stuff model (\hat{p}^{S} in the paper).
%
% corresp: the inferred correspondence from each cell in the query image to the cells in the predictor 
%          images (f from the paper).  Each cell of corresp has the following fields:
%            - mu: the i'th column is the mean of the inferred distribution over HOG cells in the i'th
%                  predictor image.
%            - covar: the i'th column is the covariance of the same distribution (vectorized via (:)).
%            - level: the level of the HOG pyramid that mu and covar refer to in a given predictor image.
%                     It indexes into the returned pyrs structure, which is computed on a double-scale image,
%                     i.e. The size of this corresponding level will be 
%                     (floor(round(size(predictorimage_i)*2.^(-(level(i)-9)/levels_per_octave))/sBins)-2) 
%
% pyrs: a cell array where the i'th cell contains the pyramids constructed for the i'th predictor images 
%       (i.e. predictorbbox(i,:)) via constructFeaturePyramidForImg.  The only nonempty element of 
%       pyra{i}.features will be the one used for prediction.
%
% inimpyrlevel: the level of the HOG pyramid constructed for the query image that was used.
%
% blurthingprob: The pseudo-bayesian estimate of the probability that each cell is 'thing' (c from the paper).
%
% exempconfidence: The pseudo-bayesian estimate of the probability that a cell in the query image corresponds
%                  to each of the predictor images (\omega from the paper).  
%                  size(exempconfidence)==[size(thingloglik) size(predictorbbox,1)].
%
% mimicryscore: the probability computed by the foreground model that each cell in the query image is thing
%               before it predicts the cell, which is used to determine whether the algorithm should
%               mimic the background model (\beta*c from the paper).  This is also set to zero for cells
%               that do not contain enough gradient energy.
%
% orig_patch_loc: the initial condition region; this is approximately equal to the cells in the image that 
%                 would have been used to obtain the initial bounding box in querybbox.
%
% mimicryscorenooc: the mimicry score without multiplying in blurthingprob (which is supposed to model occlusion).
%                   (\beta from the paper).  Yhis is set to 0 for cells that do not have enough gradient
%                   energy.
%           

function [thingloglik, stuffloglik, corresp,pyrs,inimpyrlevel,blurthingprob,exempconfidence,mimicryscore,orig_patch_loc,mimicryscorenooc]=contextpredict(querybbox,predictorbbox,stuffmodel,confidencemaps,conf)
  global ds;
  try
  if(~exist('conf','var'))
    conf=struct();
  end
  if(dsfield(ds,'conf','params'))
    conf=overrideConf(ds.conf.params,conf);
  end
  defaultparams=struct(...
    'scaleIntervals', 8,...
    'sBins', 8,...
    'levelFactor', 2,...
    'lambda',.5,...
    'lambdaprime',2.5,...
    'maxpreds',Inf,...
    'maxcomptime',Inf,...
    'maxinf',200)
  conf=overrideConf(defaultparams,conf);
  if(size(querybbox,2)<7)
    querybbox(:,7)=0;
  end
  if(size(querybbox,2)<8)
    querybbox(:,8)=0;
  end
  if(size(predictorbbox,2)<7)
    predictorbbox(:,7)=c(1:size(predictorbbox,1));
  end
  if(size(predictorbbox,2)<8)
    predictorbbox(:,8)=0;
  end
  comptimer=tic;
  % Remove one cell along the border of the patch.  These cells aren't used
  % during the nearest neighbors phase; they are only included in the 
  % bounding box to make sure that we can normalize every HOG cell in the patch
  % representation correctly based only on the patch.
  cropcells=(conf.patchCanonicalSize(1)-2*conf.sBins)/conf.patchCanonicalSize(1);
  querybbox=scaledets(querybbox,cropcells);
  predictorbbox=scaledets(predictorbbox,cropcells);
  cellsperbox=8;

  % Extract the HOG pyramid for the query image.
  bbox2hogim=[];
  if(isfield(conf,'queryimage'))
    bbox2hogim=conf.queryimage;
  end
  [hogpyr,idx]=bbox2hog(querybbox,cellsperbox,bbox2hogim,conf);
  ctrpos=[idx(2) idx(1)]+cellsperbox/2-1;
  inimpyrlevel=idx(end);
  hogim=hogpyr.features{idx(end)};

  % Figure out which cells have enough gradient energy.  This is based
  % on both the cells and their neighbors; hence the convolution.  This is
  % necessary because cells with very low gradients tend to be dominated
  % by JPEG artifacts after HOG's aggressive normalization.  These artifacts 
  % are structured in a way that the thing model tends to like.
  sufficientgradient=(conv2(hogpyr.gradimg{idx(end)},ones(3,3),'same')>9);
  clear hogpyr;

  % The most expensive computation in the algorithm is equation 9.  As written
  % in the paper, though, it's very inefficient.  Here's a speedup.  We compute
  % \mathcal{N}(H^{i}_{u,v};H^{0}_{x,y},\Sigma_H) as 
  % C*exp(-(H^{0}_{x,y}-H^{i}_{u,v})*\Sigma_H^-1*(H^{0}_{x,y}-H^{i}_{u,v})'/2).  The term in the exponent
  % can be written as:
  %
  % H^{0}_{x,y}*\Sigma_H^-1*H^{0}_{x,y}' (which is a constant that we store in hogim_precompute(:,:,end))
  % -2*H^{0}_{x,y}*\Sigma_H^-1*H^{i}_{u,v}' (we store \Sigma_H^-1*H^{i}_{u,v} in pyrs2{i}(:,:,1:end-1))
  % +H^{i}_{u,v}*\Sigma_H^-1*H^{i}_{u,v}' (this term also occurs in the second term of equation 9, and
  % since we only care about the ratio between the first and second terms, it can be ignored.)
  %
  % Furthermore, \mathcal{N}(H^{i}_{u,v};\mu_H,\Sigma_H) 
  % can be written as 
  %
  % C*exp(-(\mu_H-H^{i}_{u,v})*\Sigma_H^-1*(\mu_H-H^{i}_{u,v})'/2).  Removing the same terms
  % that we removed above, we can simplify this to
  % exp(-(\mu_H*\Sigma_H^-1*\mu_H'-2*H^{i}_{u,v}*\Sigma_H^-1*\mu_H)/2)
  % which we store in pyrs2{i}(:,:,end)
  %
  % In this way, we can reduce an O(d^2) computation to O(d), where, d is the HOG dimension.
  %
  % We also permute the dimensions of the HOG data that gets passed to the warping function,
  % so that each HOG cell is contiguous in memory.
  bgcovar=stuffmodel.covar+eye(size(hogim,3))*.001;
  bgmu=stuffmodel.mu;
  hogim_precompute=hogim;
  hogim_precompute(:,:,end+1)=0;
  for(y=1:size(hogim_precompute,1))
    for(x=1:size(hogim_precompute,2))
      hogim_precompute(y,x,end)=c(hogim(y,x,:))'*(bgcovar\c(hogim(y,x,:)));
    end
  end
  hogim_precompute=permute(hogim_precompute,[3 1 2]);
  idx(end)=1;
  pyridxs=zeros(size(predictorbbox,1),3);
  for(i=1:size(predictorbbox,1))
    bbox2hogim=[];
    if(isfield(conf,'predictorimages'))
      bbox2hogim=conf.predictorimages{i};
      predictorbbox(i,7)=i;
    end
    [pyrs{i},pyridxs(i,:)]=bbox2hog(predictorbbox(i,:),cellsperbox,bbox2hogim,conf);
    exempsuffgrad{i}=(conv2(pyrs{i}.gradimg{pyridxs(i,end)},ones(3,3),'same')>9);
  end
  for(i=1:size(predictorbbox,1))
    pyrs2{i}=pyrs{i};
    for(k=1:numel(pyrs{i}.features))
      if(isempty(pyrs{i}.features{k}))
        continue;
      end
      sz=size(pyrs{i}.features{k});
      data=(reshape(pyrs{i}.features{k},[],size(hogim,3)))*inv(bgcovar);
      pyrs2{i}.features{k}=-reshape(data,sz(1),sz(2),sz(3))*2;
      pyrs2{i}.features{k}(:,:,end+1)=reshape(exp(-(-2*data*c(bgmu)+c(bgmu)'*(bgcovar\c(bgmu)))/2),sz(1),sz(2));
      pyrs2{i}.features{k}=permute(pyrs2{i}.features{k},[3 1 2]);
    end
  end
  
  % Initialize corresp, which is our representation of f, plus a bunch of
  % other variables to keep track of our predictions.
  [corresp]=initcorresp(hogim,idx,[cellsperbox cellsperbox],pyrs,pyridxs);
  thingloglik=zeros(size(hogim(:,:,1)));
  stuffloglik=zeros(size(hogim(:,:,1)));
  have_preds=~cellfun(@isempty,corresp);
  orig_patch_loc=have_preds;
  mimicryscore=double(have_preds);
  mimicryscorenooc=double(have_preds);

  % some slightly more tricky initialization.  We initialize
  % the thingprob and stuffprob so that we have .5 probability
  % for the initial condition region.  This value gets used
  % in generating the mimicry score; by setting it to .5, the
  % initial c (i.e. blurthingprob) values will all be .5.
  thingprob=zeros(size(have_preds));%this value is never read
  thingprob(have_preds(:))=.5;
  % stiffprob is always equal to 1-thingprob, but it's important
  % to store them separately in case thingprob gets very close
  % to 1; we dont' want to take the log of 0.
  stuffprob=thingprob;
  % By setting these values to 1/size(predictorbbox,1), we trust
  % all predictor images equally initially.
  exempaccuracy=repmat(zeros(size(have_preds)),[1,1,numel(pyrs)]);%this value is never read.
  exempaccuracy(repmat(have_preds(:),size(predictorbbox,1),1))=1/size(predictorbbox,1);
  % we keep 1-exempaccuracy separate so that we can represent very small numbers
  % without running out of precision.
  onemexempaccuracy=1-exempaccuracy;

  % The optimizecorresp mex file has a cache that is stateful; 
  % since we're starting a new image, we tell the mex file to clear it.
  clearcache=true;

  % which predictor image to show the warping for it conf.disp=1. 
  if(isfield(conf,'imtodisp'))
    imtodisp=conf.imtodisp;
  else
    imtodisp=1;
  end
  npredits=-1;
  while(true)
    npredits=npredits+1;
    have_preds_prev=have_preds;
    a=tic

    % equation 8 from the paper
    tmp=exp(gaussfiltervalid(log(thingprob),have_preds,2));
    blurthingprob=tmp./(tmp+exp(gaussfiltervalid(log(stuffprob),have_preds,2)));
    if(any(isnan(blurthingprob(have_preds))))
      error('nans in blurthingprob');
    end

    % equation 26 from the paper
    exempconfidence=repmat(thingprob*0,[1,1,size(predictorbbox,1)]);
    for(i=1:size(exempconfidence,3))
      tmp=exp(gaussfiltervalid(log(exempaccuracy(:,:,i)),have_preds,2));
      exempconfidence(:,:,i)=tmp./(tmp+exp(gaussfiltervalid(log(onemexempaccuracy(:,:,i)),have_preds,2)));
    end
    exempconfidence=capprobabilities(exempconfidence,1/3,3);

    % Decide which cells to predict using the current condition region.  In practice, 
    % we make a minor approximation
    % to the paper here, by inferring multiple cells in parallel. That is, we actually 
    % use the same
    % condition region for multiple prediction regions, which is mathematically not
    % correct.  This is necessary because optimizecorresp is so slow that we'd rather
    % not call it more than absolutely necessary.  In practice this does mean
    % that our estimates of the overall likelihood of the patch will be wrong (in
    % general, the estimator's variance increases), because the values for
    % for cells inferred simultaneously should actually provide information about
    % each another.  However, this inaccuracy is not so bad as to break the algoithm,
    % since we only allow ourselves to do a few cells in parallel.
    %
    % The cells that get inferred on a particular round are those which (a) have
    % at least one neighbor that's already been predicted, and (b) are near to
    % cells which the thing model expected to be foreground.  We also enforce
    % that the condition region remain convex, which is why we take the.
    % convex hull.  There's almost certainly better ways to do this, but
    % this was the first thing I thought of :-/
    [y2,x2]=find(mimicryscore>.05);
    hull=convhull(x2,y2,'simplify',true);
    hull=expandPolygon([x2(hull),y2(hull)],1.1);
    [y3,x3]=find(have_preds);
    x3=[x3;hull(:,1)];
    y3=[y3;hull(:,2)];
    hull=convhull(x3,y3,'simplify',true);
    hull=[x3(hull) y3(hull)];
    [yround,xround]=find(imdilate(have_preds,[0 1 0; 1 1 1; 0 1 0])-have_preds);
    valid=inpolygon(xround,yround,hull(:,1),hull(:,2));
    yround=yround(valid);xround=xround(valid);

    % if we didn't find anything that the algorithm thinks might be foreground,
    % or we've run out of our computation budget, we return.
    if(isempty(xround)||(sum(have_preds(:))-sum(orig_patch_loc(:))>=conf.maxpreds)||(toc(comptimer)>conf.maxcomptime))
      mimicryscore(find(orig_patch_loc))=0;
      return
    end

    % One additional hack that makes things work a little better is to make the
    % algorithm trust the features near the boundry more than it trusts those near
    % far from the boundary, since that helps us better align the contours where
    % they need to be predicted.  We also want to prevent the confidence value
    % from going to zero, since that will prevent the algorithm from getting back
    % on track after, e.g., an occlusion.
    confidencetmp=bsxfun(@times,max(.1,blurthingprob),max(1,4-bwdist(~have_preds)).*sufficientgradient);

    % run the optimizer.
    [corresp,transf]=optimizecorrespwrap(hogim_precompute,corresp,pyrs2,conf.maxinf*numel(xround),confidencetmp,struct('lambda',conf.lambda,'lambdaprime',conf.lambdaprime,'clearcache',clearcache));
    clearcache=false;

    if(exist('stopfile','file'))
      keyboard
    end
    disp(['inferring correspondence: ' num2str(toc(a))]);

    % noe generate the stuff predictions
    b=tic;
    [stuffmus,stuffcovars]=stuffmodelpredict(hogim,have_preds,xround,yround,stuffmodel);
    for(i=1:numel(xround))
      topredict=c(hogim(yround(i),xround(i),:))';
      mu=stuffmus(:,:,i);
      sigma=stuffcovars(:,:,i);
      stuffloglik(yround(i),xround(i))=-log(det(sigma))/2-(topredict-c(mu)')*(sigma\(c(topredict)-c(mu)))/2;
    end
    disp(['stuffmodel ' num2str(toc(b))])
    
    % Generate the predictions based on the estimated correspondence.
    for(infpos=1:numel(xround))

      % First find a neighbor in the conditio nregion
      neighbors=[0 1; 0 -1; 1 0; -1 0];
      x=xround(infpos);
      y=yround(infpos);
      for(i=1:size(neighbors,1))
        if(valididx(corresp,y+neighbors(i,1),x+neighbors(i,2))&&~isempty(corresp{y+neighbors(i,1),x+neighbors(i,2)}))
          break;
        end
      end
      neighbor=[y x]+neighbors(i,:);

      % Generate a new correspondence estimate for the current prediction
      % region by extrapolating
      mycorresp=corresp{neighbor(1),neighbor(2)};
      mycorresp.mu=bsxfun(@minus,mycorresp.mu,neighbors(i,[2 1])');
      corresp{y,x}=mycorresp;
      
      % Now we aggregate the data over the predictor images.  We do this
      % simultaneously to make the prediction (aggregating over HOG cells)
      % and to estimate the mimicry score (\beta in the paper), since they
      % need to aggregate over the same region.
      transferredfeats={};
      wts={};
      pats={};
      topredict=c(hogim(y,x,:))';
      betanormfact=0;
      transferredbeta=0;

      d=tic;
      % for each predictor image:
      for(curidx=1:size(mycorresp.mu,2))
        % first extract a high-probability region so we don't have to 
        % integrate everything.  We do this by finding the major
        % and minor axes of the gaussian and creating a box that fits both
        % of them.
        currcovar=reshape(mycorresp.covar(:,curidx),[2,2]);
        [V,D]=eig(currcovar);
        dist=sqrt(abs((log(.0001)+log(det(currcovar))/2)/(V(:,1)'*inv(currcovar)*V(:,1)/2)));
        dist2=sqrt(abs((log(.0001)+log(det(currcovar))/2)/(V(:,2)'*inv(currcovar)*V(:,2)/2)));
        V=max(abs(dist*V(:,1)),abs(dist2*V(:,2)));
        if(any(V==0))
          disp('covar shrank!');
          keyboard
        end
        windowx=floor(corresp{y,x}.mu(1,curidx)-V(1)):ceil(corresp{y,x}.mu(1,curidx)+V(1));
        windowy=floor(corresp{y,x}.mu(2,curidx)-V(2)):ceil(corresp{y,x}.mu(2,curidx)+V(2));
        [gridx,gridy]=meshgrid(windowx,windowy);
        pts=[gridx(:)';gridy(:)'];
        pts=bsxfun(@minus,pts,corresp{y,x}.mu(:,curidx));
        probs=reshape(1/(2*pi*sqrt(det(currcovar)))*exp(-sum(pts.*(inv(currcovar)*pts),1)/2),size(gridx));
        probs=probs/sum(probs(:));

        % trim the window to fit inside the predictor image.
        validy=(windowy>0&windowy<=size(pyrs{curidx}.features{corresp{y,x}.level(curidx)},1));
        validx=(windowx>0&windowx<=size(pyrs{curidx}.features{corresp{y,x}.level(curidx)},2));
        probs=probs(validy,validx);
        outprobs=1-sum(probs(:));
        windowy=windowy(validy);
        windowx=windowx(validx);

        % finally extract the corresponding features, and use the gaussian probabilities as weights.
        % Discard cells that don't have enough gradient energy.
        transferredfeats{curidx,1}=reshape(pyrs{curidx}.features{corresp{y,x}.level(curidx)}(windowy,windowx,:),[],size(hogim,3));
        probs(~exempsuffgrad{curidx}(windowy,windowx))=[];
        wts{curidx,1}=(probs(:))*exempconfidence(y,x,curidx);
        transferredfeats{curidx}(~exempsuffgrad{curidx}(windowy,windowx),:)=[];

        % finally, aggregate statistics to compute \beta.
        if(~isempty(confidencemaps{curidx}))
          % if the confidence map that was passed in isn't the right size (e.g. it was computed on a
          % slightly different HOG pyramid), resize it.
          if(~all(size(pyrs{curidx}.features{corresp{y,x}.level(curidx)}(:,:,1))==size(confidencemaps{curidx})))
            confidencemaps{curidx}=imresize(padarray(confidencemaps{curidx},1),size(pyrs{curidx}.features{corresp{y,x}.level(curidx)}(:,:,1))+2,'bilinear');
            confidencemaps{curidx}=confidencemaps{curidx}(2:end-1,2:end-1);
          end
          cmwind=confidencemaps{curidx}(windowy,windowx);
          cmwind(~exempsuffgrad{curidx}(windowy,windowx))=[];
          transferredbeta=transferredbeta+sum(sum(probs.*cmwind))+.5*outprobs;
          betanormfact=betanormfact+1;
        end
      end

      % Now that we have aggregated the confidence (g) values for all predictor images,
      % we can compute the final mimicry score (\beta*c from the paper).
      if(betanormfact>0)
        mimicryscore(y,x)=transferredbeta/betanormfact;
      else
        mimicryscore(y,x)=.5;
      end
      mimicryscorenooc(y,x)=mimicryscore(y,x).*sufficientgradient(y,x);
      mimicryscore(y,x)=mimicryscore(y,x).*blurthingprob(y,x).*sufficientgradient(y,x);

      % Aggregate all HOG cells that the current HOG cell might correspond to.  If there's
      % nothing we correspond to (i.e. we're outside the bounds of all predictor images),
      % then just mimic the stuff model.  Then make our thing model prediction.
      if(sum(cellfun(@numel,wts))~=0)
        [mu,sigma]=weightedgaussian(cell2mat(transferredfeats),cell2mat(wts));
        sigma=sigma+eye(size(sigma,2))*.001;%HOG is not actually linearly independent
        thingloglik(y,x)=-log(det(sigma))/2-(topredict-c(mu)')*(sigma\(c(topredict)-c(mu)))/2;
      else
        sigma=bgcovar;
        thingloglik(y,x)=stuffloglik(y,x);
      end

      % Now make the 'mixture of experts' predictions to figure out which are the
      % good predictor images (eq. 14 from the paper)
      for(curidx=1:size(mycorresp.mu,2))
        if(isempty(wts{curidx}))
          exempaccuracy(y,x,curidx)=1e-10;
        else
          tmpmu=(wts{curidx}'./sum(wts{curidx}))*transferredfeats{curidx};
          tmpprob=-log(det(sigma))/2-(topredict-c(tmpmu)')*(sigma\(c(topredict)-c(tmpmu)))/2;
          exempaccuracy(y,x,curidx)=(tmpprob);
        end
      end
      exempaccuracy(y,x,:)=exp(exempaccuracy(y,x,:))/sum(exp(exempaccuracy(y,x,:)));
      for(k=1:size(exempaccuracy,3))
        onemexempaccuracy(y,x,k)=sum(exp(exempaccuracy(y,x,[1:k-1,k+1:end])))/sum(exp(exempaccuracy(y,x,:)));%*.9+.1/size(exempaccuracy,3);
      end

      thingprob(y,x)=exp(thingloglik(y,x))/(exp(thingloglik(y,x))+exp(stuffloglik(y,x)));
      stuffprob(y,x)=exp(stuffloglik(y,x))/(exp(thingloglik(y,x))+exp(stuffloglik(y,x)));
    end
    have_preds(sub2ind(size(have_preds),yround(:),xround(:)))=true;
    disp(['thingmodel predict:' num2str(toc(d))]);
    disp([num2str(sum(have_preds(:))-sum(orig_patch_loc(:))) '/' num2str(numel(have_preds)-sum(orig_patch_loc(:)))]);

    % This block of code has no effect other than generating displays.
    if(dsbool(conf,'disp')&&mod(npredits,1)==0&&~dsbool('ds','sys','distproc','mapreducer'))
      if(isfield(conf,'queryimage'))
        inim=conf.queryimage;
        if(querybbox(1,8))
          inim=inim(:,end:-1:1,:);
        end
      else
        inim=getimg(querybbox(1,7:8));
      end

      hfig=sfigure(1);
      figpos = get(hfig, 'Position');
      set(hfig, 'Position', [figpos(1), figpos(2), 1024, 512]);
      subplot(2,3,4);
      disploglik=myimagesc(thingloglik-stuffloglik,[-20 20]);
      disploglik(repmat(~have_preds|orig_patch_loc,[1 1 3]))=1;
      imagesc(padarraycolor(disploglik,1,[1 1 1]));
      axis equal;
      title('log(p^{T}/p^{S})');
      if(isfield(conf,'predictorimages'))
        im=conf.predictorimages{imtodisp};
      else
        im=getimg(predictorbbox(imtodisp,7));
      end
      if(predictorbbox(imtodisp,8))
        im=im(:,end:-1:1,:);
      end
      imwarp=correspwarp(corresp,conf.sBins,im,pyrs,imtodisp);
      subplot(2,3,1);
      dispinim=im2double(imresize(inim,2^(-(inimpyrlevel-1-conf.scaleIntervals)/conf.scaleIntervals)));
      sz=conf.sBins*floor(size(dispinim(:,:,1))/conf.sBins);
      dispinim2=dispinim(1:sz(1),1:sz(2),:);
      imwarp=imwarp(1:sz(1),1:sz(2),:);
      imagesc(dispinim2)
      title('query image');
      axis equal;
      subplot(2,3,5);
      imagesc(imwarp);axis equal;
      title('estimated correspondence')
      subplot(2,3,2);
      imagesc(min((imwarp+dispinim(1:size(imwarp,1),1:size(imwarp,2),:))/2,1));axis equal;
      title('estimated correspondence')
      subplot(2,3,3);
      dispmimicry=myimagesc(mimicryscorenooc);
      dispmimicry(repmat(~have_preds|orig_patch_loc,[1 1 3]))=1;
      imagesc(padarraycolor(dispmimicry,1,[1 1 1]));axis equal;
      title('mimicry score ($\beta$)','interpreter','latex')
      subplot(2,3,6);
      dispbayes=myimagesc(blurthingprob);
      dispbayes(repmat(orig_patch_loc|~imdilate(have_preds,fspecial('disk',3)>.004),[1 1 3]))=1;
      imagesc(padarraycolor(dispbayes,1,[1 1 1]));
      axis equal;
      title('bayesian estimate (c_{x,y})')
      hfig=sfigure(3)
      clf
      figpos = get(hfig, 'Position');
      set(hfig, 'Position', [figpos(1), figpos(2), 1920, 300]);
      axis equal;
      [~,ord]=sort(c(nansum(nansum(bsxfun(@times,exempconfidence,thingprob),1),2)),'descend')
      for(k=1:min(12,size(predictorbbox,1)))
        subplot(2,12,k)
        if(isfield(conf,'predictorimages'))
          predrim=conf.predictorimages{ord(k)};
          if(predictorbbox(ord(k),8))
            predrim=predrim(:,end:-1:1,:);
          end
          imagesc(predrim)
        else
          imagesc(getimg(predictorbbox(ord(k),7:8)));
        end
        title(' ');
        set(gca,'XTick',[])
        set(gca,'YTick',[])
        axis equal;
      end
      for(k=1:min(12,size(exempconfidence,3)))
        subplot(2,12,12+k)
        data=myimagesc(exempconfidence(:,:,ord(k)).*have_preds,[0 .3]);
        data(repmat(~have_preds|orig_patch_loc,[1 1 3]))=1;
        imagesc(padarraycolor(data,1,[1 1 1]));axis equal;
        set(gca,'XTick',[])
        set(gca,'YTick',[])
      end
      tightfig;
      ha = axes('Position',[0 0 1 1],'Xlim',[0 1],'Ylim',[0 1],'Box','off','Visible','off','Units','normalized', 'clipping' , 'off');
      text(0.5, 1,'Ranked list of predictor images in terms of contribution to the prediction','HorizontalAlignment'      ,'center','VerticalAlignment', 'top','interpreter','latex')
      text(0.5, .5,'The region in query image where the predictor image above predicted well ($\omega^{i}_{\mathcal{C}[t]}$)','HorizontalAlignment'      ,'center','VerticalAlignment', 'top','interpreter','latex')
    end

    % Optionally also generate a display of the alpha's from the
    % correspondence algo (stored in transf).
    if(false && dsbool(conf,'disp'))
      sfigure(5)
      for(k=1:4)
        subplot(2,2,k)
        todisp=transf{imtodisp}(:,:,ceil(k/2),mod(k-1,2)+1);
        if(k==1||k==4)
          todisp=todisp-have_preds_prev;
        end
        imagesc(padarray(todisp,[1,1]),[-1 1]);
      end
    end

    drawnow;
  end

  catch ex,dsprinterr;end
  
end

% Given a square bounding box ([x1 y1 x2 y2]), generate
% a HOG feature where that bounding box is approximately
% ncells tall.  pyr will be a full pyramid structure
% generated by constructFeaturePyramidForImg, but only one
% level of pyr.features will be filled in.  idx
% gives the coordinates of the upper-right corner cell of the
% bounding box in the HOG representation.  That cell may
% be referenced via pyr.features{idx(3)}(idx(1),idx(2)).
function [pyr,idx]=bbox2hog(bbox,ncells,im,conf)
  if(isempty(im))
    im=getimg(bbox(7));
  end
  im=imresize(im,2);
  bbox(1:4)=(bbox(1:4)-.5)*2+.5
  if(bbox(8))
    im=im(:,end:-1:1,:);
    bbox([1 3])=-bbox([3 1])+size(im,2)+1;
  end
  scale=(ncells)*conf.sBins/(bbox(3)-bbox(1)+1);
  scale2=max(1,round(conf.scaleIntervals*-log2(scale))+1);
  pyr=constructFeaturePyramidForImg(im2double(im),conf,scale2);
  hogim=pyr.features{scale2};
  realscale=size(im(:,:,1))./round(size(im(:,:,1))/pyr.scales(scale2));%floor(size(im(:,:,1))/conf.sBins)./(size(pyr.features{scale2}(:,:,1))+2);i
  idx=(((bbox(1:2)-.5)./[realscale([2 1])])/conf.sBins);
  %keyboard
  pyr.scales=pyr.scales/2;
  idx=round(idx([2 1]));
  %idx=(((bbox(1:4)-1)./[realscale([2 1]) realscale([2 1])])/conf.sBins)-1;
  idx=max(1,min(idx,size(hogim(:,:,1))-ncells+1));
  idx=round([idx scale2]);
end
