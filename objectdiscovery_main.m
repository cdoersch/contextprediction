% This is the main script for the full object discovery pipeline, including
% nearest neighbors, training the stuff model, running context prediction for
% object discovery, and transferring car keypoints.  
%
% Before running the code, there are three things you have to set up, listed
% as STEP 1-3 over the next few lines.

global ds;
myaddpath;

% STEP 1: Configure a global output directory; must be shared among all workers.
dssetout('/SOME/ABSOLUTE/PATH');

% In case we're restarting but cd'd into some directory
dscd('.ds');

% STEP 2: Distributed processing setup: host to submit jobs to (or 'local' if you
% don't want to use qsub), the number of parallel workers, and any flags
% to be used when calling qsub.  For this code, I recommend one physical 
% core per worker.  See dsmapredopen for details.
targmachine='local'
njobs=12;
qsubopts=''
ds.conf.mempermachine=59; % Number of GB of RAM on each machine.  Necessary so we don't try to 
                          % run too many jobs simultaneously.  Minimum 8GB
                          % for learning the GMM.

% STEP 3: Initialize PASCAL dataset into ds.conf.gbz{22} and ds.imgs{22}.  Set the following
% to the patch contaiining the VOC2011 directory.  If you don't want to use PASCAL, see 
% setdataset.m for info on how to format the dataset descriptor.  This directory needs to 
% be accessible on all workers.
pascalpath='/SOME/ABSOLUTE/PATH/THAT/ENDS/WITH/VOC2011';
if(~exist('dataset_pascal.mat','file'))
  preprocesspascal(pascalpath,'dataset_pascal.mat');
end
pascaldata=load('dataset_pascal.mat');
weburl='';
setdataset(22,pascaldata.imgs,pascalpath,'',weburl);
dssave;
if(isfield(ds.conf.gbz{ds.conf.currimset},'imgsurl'))
  ds.imgsurl=ds.conf.gbz{ds.conf.currimset}.imgsurl;
end

rand('seed',1234)

ds.conf.params=struct( ...
  'patchCanonicalSize', {[80 80]}, ... % Size of patch to use during initial nearest neighbors phase.
                                   ... % Should be square.
  'scaleIntervals', 8,             ... % Levels per octave in HOG pyramid
  'sBins', 8,                      ... % HOG cell size in pixels
  'levelFactor', 2,                ... % Downsampling factor for each 'octave' of the HOG pyramid.  Leave at 2.
  'whitening', 1,                  ... % Use whitening during nearest neighbors phase.
  'normbeforewhit', 1,             ... % Normalize features before whitening them.
  'normalizefeats', 1,             ... % Re-normalize features after whitening them.
  'includeflips', 1,               ... % Include flipped patches
  'samplingOverlapThreshold', 0.6 ... % patches sampled initially can't have overlap larger
                                   ... % than this value.
)

ds.conf.lambda=.1;      % Lambda parameter for warping: controls stiffness of warp
ds.conf.lambdaprime=.5;   % Lambda' parameter for warping: controls smoothness of warp
ds.conf.maxcomptime=120; % Max time allowed for each patch verification, in seconds.

imgs=ds.imgs{ds.conf.currimset};

% which images should we find nearest neighbors in?
ds.myiminds=1:numel(imgs.fullname);
% which images should we sample patches from?

% start the distributed session
dsload('ds.singlecompthread')
dsload('ds.finishedconditionallearn')
if((~dsmapredisopen()||dsbool(ds,'singlecompthread'))&&~dsbool(ds,'finishedconditionallearn'))
  if(dsmapredisopen())
    dsmapredclose;
    dsup('ds.singlecompthread',false);
  end
  distprocconf=struct();
  distprocconf.qsubopts=qsubopts;
  dsmapredopen(njobs,targmachine,distprocconf);
end

% Compute the whitening matrix by aggregating sufficient statistics from a big dataset.
dsload('ds.finishedaggcov');
if(~dsbool(ds,'finishedaggcov'))
  dsdelete('ds.aggcov');
  dsdelete('ds.datamean');
  dsdelete('ds.stuffmodelgmm.mu');
  dsdelete('ds.stuffmodelgmm.covar');
  dsdelete('ds.invcovmat');
  dsdelete('ds.whitenmat');
  dsdelete('ds.finishedsampling');
  rp=randperm(numel(ds.myiminds));
  ds.aggcov.myiminds=ds.myiminds(rp(1:min(numel(rp),1500)));;
  dssave;
  dscd('ds.aggcov');
  dsrundistributed('aggregate_covariance',{'ds.myiminds'},struct('allatonce',1,'noloadresults',1,'maxperhost',8,'waitforstart',1));
  %end
  total=0;
  clear featsum dotsum;
  dsload('ds.n');
  for(i=1:numel(ds.n))
    if(isempty(ds.n{i})),continue;end
    total=total+dsload(['ds.n{' num2str(i) '}'],'clear');
    if(~exist('dotsum','var'))
      dotsum=dsload(['ds.dotsum{' num2str(i) '}'],'clear');
    else
      dotsum=dotsum+dsload(['ds.dotsum{' num2str(i) '}'],'clear');
    end
    if(~exist('featsum','var'))
      featsum=dsload(['ds.featsum{' num2str(i) '}'],'clear');
    else
      featsum=featsum+dsload(['ds.featsum{' num2str(i) '}'],'clear');
    end
    if(any(isnan(dotsum(:)))||any(isnan(featsum(:))))
      keyboard;
    end
    disp(i);
  end
  covmat=(dotsum./total-(featsum'./total)*(featsum./total));
  covmat=covmat+.01*eye(size(covmat,1));
  dscd('.ds');
  ds.datamean=featsum./total;
  hogsz=numel(ds.datamean)./prod(ds.conf.params.patchCanonicalSize/ds.conf.params.sBins-2);
  % These values are used as \mu_H and \Sigma_H in eq. 9
  %ds.stuffmodelgmm.mu=ds.datamean(1:numel(ds.datamean)/hogsz:end);TODO remove
  %ds.stuffmodelgmm.covar=covmat(1:numel(ds.datamean)/hogsz:end,1:numel(ds.datamean)/hogsz:end);
  disp('performing matrix square root...');
  ds.invcovmat=inv(covmat);
  ds.whitenmat=sqrtm(ds.invcovmat);
  clear featsum dotsum total;
  clear covmat;
  dsdelete('ds.aggcov');
  ds.finishedaggcov=true;
  dssave;
end

% Sample random patches from the dataset to serve as seeds for the clusters.
% This code is taken almost verbatim from the discriminative mode seeking work.
dsload('ds.finishedsampling');
if(~dsbool(ds,'finishedsampling'))
  disp('sampling positive patches');
  dsdelete('ds.sample');
  dsdelete('ds.detectors');
  dsdelete('ds.initPatches');
  dsdelete('ds.batchfordetr');
  dsdelete('ds.finisheddetection');

  initFeatsExtra=[];
  initPatsExtra=[];

  ds.sample=struct();
  ds.sample.initInds=ds.myiminds;
  dsrundistributed('[ds.sample.patches{dsidx}, ds.sample.feats{dsidx}]=sampleRandomPatchesbb(ds.sample.initInds(dsidx),1,struct());',{'ds.sample.initInds'},struct('maxperhost',ds.conf.mempermachine/4));

  % divide the sampled patches into batches (two images' worth of sampled
  % patches per batch). The batches are purely for efficiency, specifically
  % to limit the number of files that get written during the nearest neighbors phase. 
  batch_size=40;
  allpatches=cell2mat(ds.sample.patches(:));
  allfeats=cell2mat(ds.sample.feats(:));
  ids=(1:size(allfeats,1))';
  batchpartition=genbatches(numel(ids),batch_size);%c(bsxfun(@times,ones(batch_size,1),1:ceil(size(allpatches,1)/batch_size)));
  [allpatches,allfeats,ids]=distributeby(allpatches,allfeats,ids,batchpartition(1:size(allpatches,1)));

  disp(['sampled ' num2str(size(allpatches,1)) ' patches']);

  % convert the patch features for each batch into a detector structure.
  ds.detectors=cellfun(@(x,y,z) struct('w',x,'b',y,'id',z),...
                     allfeats,...
                     cellfun(@(x) ones(size(x,1),1),allpatches,'UniformOutput',false),...
                     ids,...
                     'UniformOutput',false)';
  % Keep the initial sampled patches; sometimes useful for debugging.
  initPatches(1:numel(cell2mat(ids)),6)=(1:numel(cell2mat(ids)))';
  ds.initPatches=initPatches;

  %Lookup table for each detector's batch.
  ds.batchfordetr=[cell2mat(ids) cell2mat(cellfun(@(x,y) x*0+y,ids,c(num2cell(1:numel(ids))),'UniformOutput',false))];
  ds.finishedsampling=true;
  dssave();
end


dsload('ds.finisheddetection')
if(~dsbool(ds,'finisheddetection'))
  dsdelete('ds.untraineddets')
  dsdelete('ds.topnuntrained')
  dsdelete('ds.finishedgmmlearn');
  dsmapreduce(['detectors=effstrcell2mat(dsload(''ds.detectors'')'');'...
               '[dets]=detectInIm(detectors,ds.myiminds(dsidx),struct(''multperim'',false,''flipall'',true));'...
               'ctridx=dsload(''ds.batchfordetr'');'...
               'if(~isempty(dets)),'...
                 '[~,ctrpos]=ismember(dets(:,6),ctridx(:,1));'...
                 '[dets,outpos]=distributeby(dets,ctridx(ctrpos,2));'...
                 'ds.untraineddets(outpos,dsidx)=dets;'...
               'end'],...
               ['dets=cell2mat(ds.untraineddets(dsidx,:)'');'...
               'dets=distributeby(dets,dets(:,6));'...
               'ds.topnuntrained{dsidx}=cell2mat(maxkall(dets,5,200));'],'ds.myiminds','ds.untraineddets',struct('noloadresults',1),struct('maxperhost',floor(ds.conf.mempermachine/8)));
   ds.finisheddetection=true;
   dssave;
end

%end

dsload('.ds.finishedgmmlearn')
if(~dsbool(ds,'finishedgmmlearn'))
  dsdelete('ds.finishedconditionallearn');
  dscd('.ds.genstuffmodel');
  dsdelete('ds.genlibrary');
  dsdelete('ds.stuffmodelgmm');

  % Generate the shape of each of our GMM's.  We use 9x9 diamonds cut in half.
  halves=genhalves(9);

  % When generating the patch library, we want 9-by-9-cell patches and
  % we don't want them to be whitened.  Internally, this uses the same
  % unentanglePyramid function as the detection phase, so we need to
  % update the settings.
  dsup('ds.conf.params.patchCanonicalSize',[(size(halves(:,:,1))+2)*ds.conf.params.sBins]);
  dsup('ds.conf.params.whitening',0);
  dsup('ds.conf.params.normalizefeats',0);
  dsup('ds.conf.params.normbeforewhit',0);
  dsup('ds.conf.patchsz',size(halves,1));

  imids=dsload('.ds.myiminds');
  ds.genlibrary.myiminds=imids(1:10:end);
  dssave;

  ds.stuffmodelgmm.valid=halves;
  dscd('ds.genlibrary');

  % Generate the set of 1,000,000 patches that we'll use to train our GMM's.
  dsrundistributed('genlibrarydist','ds.myiminds',struct('noloadresults',true));
  dscd('.ds.genstuffmodel');
  ctr=ceil(size(halves,1)*size(halves,2)/2);

  % Restart the workers to force Matlab to clean up memory.  Learning the GMM's is the most
  % memory intensive part of the program.
  dsmapredrestart;
  % Now we actually learn the GMM dictionary.  Each job learns one dictionary.
  dsrundistributed([...
    ...% First load the learned library, and extract only those feature positions that are
    ...% valid for the current dictionary
    'dsload(''ds.genlibrary.n'');tmppats=dsload(''ds.genlibrary.pats{1}'');halves=dsload(''ds.stuffmodelgmm.valid'');'...
    'indstokeep=repmat(c(halves(:,:,dsidx)),size(tmppats,1)/numel(halves(:,:,dsidx)),1);'...
    'data=zeros(sum(indstokeep),sum(cell2mat(ds.genlibrary.n)));'...
    'curidx=1;'...
    'for(i=1:numel(ds.genlibrary.n)),'...
      'tmppats=dsload([''ds.genlibrary.pats{'' num2str(i) ''}''],''clear'');'...
      'data(:,curidx:curidx+size(tmppats,2)-1)=double(tmppats(indstokeep,:));'...
      'curidx=curidx+size(tmppats,2);'...
    'end,'...
    'ds.genlibrary=struct();'...
    'if(dsidx==1),'...
      'onecell=data(1:sum(c(halves(:,:,dsidx))):end,:);'...
      'ds.cellwisemu=mean(onecell,2)'';'...
      'ds.cellwisecovar=cov(onecell'');'...
    'end,'...
    ...% Then actually learn the GMM's.  The paper says 5000 centers, but it turns out 2000 
    ...% works essentially as well and is faster.
    'ds.stuffmodelgmm.gmm{dsidx}=gmmlearn2(data,2000,struct(''convergence'',30));'...
    ],size(ds.stuffmodelgmm.valid,3),struct('maxperhost',floor(ds.conf.mempermachine/8)));
  dsdelete('ds.genlibrary');
  dscd('.ds');
  ds.finishedgmmlearn=true;
  dssave;
end

% Next, estimate the conditional distribution over the prediction region
% for each GMM center by assigning patches
% from the dataset to each GMM component in each dictionary.
% This works by collecting sufficient statistics associated with each GMM
% component; each GMM component has its sufficient statistics aggregated
% in parallel.  We use batches again here because there can be a large number
% of components, which can result in a huge number of files.
dsload('ds.finishedconditionallearn');
if(~dsbool(ds,'finishedconditionallearn'))
  dsdelete('ds.round*.finishedverify');
  dscd('.ds.genstuffmodel');
  dsdelete('ds.myiminds');
  dsdelete('ds.batches');
  dsdelete('ds.batchsz');
  dsdelete('ds.suffstats');
  dsdelete('ds.covmat');
  dsdelete('ds.mu');
  iminds=dsload('.ds.myiminds');
  rp=randperm(numel(iminds));
  ds.myiminds=rp(1:min(numel(rp),4000))';
  ds.batchsz=20;
  ds.batches=genbatches(sum(cellfun(@(x) size(x.ctrs,1),ds.stuffmodelgmm.gmm)),ds.batchsz);
  dsmapreduce(['dsload(''ds.stuffmodelgmm'');dsload(''ds.batches'');'...
               'ds.featsum={};ds.dotsum={};ds.ntotal={};'...
               '[dat.featsum,dat.dotsum,dat.ntotal]='...
               'incrementalgmm(ds.stuffmodelgmm,ds.myiminds(dsidx),struct(''gmm'',true));'...
               'dat=distributeby(dat,ds.batches);'...
               '[ds.suffstats(:,min(dsidx))]=dat;'...
               ],[...
               'for(j=1:numel(ds.suffstats{dsidx,1}.dotsum)),'...
                 'ntotal=0;dotsum=0;featsum=0;'...
                 'for(i=1:size(ds.suffstats,2)),'...
                   'if(~isempty(ds.suffstats{dsidx,i})&&ds.suffstats{dsidx,i}.ntotal{j}>0),'...
                     'featsum=featsum+ds.suffstats{dsidx,i}.featsum{j};'...
                     'dotsum=dotsum+ds.suffstats{dsidx,i}.dotsum{j};'...
                     'ntotal=ntotal+ds.suffstats{dsidx,i}.ntotal{j};'...
                   'end,'...
                 'end;'...
                 'if(~isfield(ds,''covmat'')||numel(ds.covmat)<dsidx||numel(ds.covmat{dsidx})<=1),'...
                   'ds.covmat{dsidx}=zeros([size(dotsum) numel(ds.suffstats{dsidx,1}.dotsum)]);'...
                   'ds.mu{dsidx}=zeros([size(featsum) numel(ds.suffstats{dsidx,1}.dotsum)]);'...
                 'end,'...
                 'ds.covmat{dsidx}(:,:,j)=dotsum/ntotal-(featsum/ntotal)''*(featsum/ntotal);'...
                 'ds.mu{dsidx}(:,:,j)=featsum/ntotal;'...
               'end'...
               ],'ds.myiminds',{'ds.suffstats'},struct('maxperhost',ds.conf.mempermachine/6),struct('allatonce',true));
end
dscd('.ds');

dsload('ds.finishedconditionallearn');
if(~dsbool(ds,'finishedconditionallearn'))
  dsload('ds.genstuffmodel.stuffmodelgmm');
  dsdelete('ds.stuffmodelgmm.gmm');
  dsdelete('ds.stuffmodelgmm.valid');
  ds.stuffmodelgmm.gmm=ds.genstuffmodel.stuffmodelgmm.gmm;
  ds.stuffmodelgmm.valid=ds.genstuffmodel.stuffmodelgmm.valid;
  for(i=1:numel(ds.genstuffmodel.mu))
    for(j=1:size(ds.genstuffmodel.mu{i},3))
      idx=(i-1)*ds.genstuffmodel.batchsz+j;
      comp=1+floor((idx-1)/size(ds.stuffmodelgmm.gmm{1}.ctrs,1));
      ctr=idx-(comp-1)*size(ds.stuffmodelgmm.gmm{1}.ctrs,1);
      ds.stuffmodelgmm.gmm{comp}.condmu(:,:,ctr)=ds.genstuffmodel.mu{i}(:,:,j);
      ds.stuffmodelgmm.gmm{comp}.condcovar(:,:,ctr)=ds.genstuffmodel.covmat{i}(:,:,j);
    end
  end
  ds.stuffmodelgmm.mu=ds.genstuffmodel.cellwisemu;
  ds.stuffmodelgmm.covar=ds.genstuffmodel.cellwisecovar;
  ds.finishedconditionallearn=true;
  dssave;
end

dsdelete('.ds.genstuffmodel');


% Optionally restart the distributed computation so that each matlab worker
% gets a single thread, which in my experience gives a speedup
% of about 10%.  Note that this does not mean each job will run
% single-threaded; by far the most expensive part of the computation
% is the warping, and inside the mex function this is multithreaded
% with OpenMP.  My suspicion is that having both OpenMP and Matlab
% multithreading just leads to too many threads and contention.
dsload('ds.singlecompthread')
if(~dsbool(ds,'singlecompthread'))
  dsmapredclose;
  if(~dsmapredisopen())
    distprocconf=struct();
    distprocconf.qsubopts=qsubopts;
    distprocconf.singleCompThread=true;
    dsup('ds.singlecompthread',true);
    dsmapredopen(njobs,targmachine,distprocconf);
  end
end

% Finally we get to the verification main loop.  At each iteration,
% we verify a certain number of patches for each cluster; the number of 
% patches verified per cluster doubles at each iteration, while the
% number of clusters is halved until we reach 1000.  Therefore, each round
% should take approximately the same time.
dsload('ds.round*.finishedverify')
if(~dsfield(ds,'round5','finishedverify'))
  if(~dsfield(ds,'round1','finishedverify'))
    dsdelete('ds.evaldets');
    dsdelete('ds.round1');
    dets=cell2mat(dsload('ds.topnuntrained')');
    evaldets=distributeby(dets,dets(:,6));
    ds.evaldets=maxkall(evaldets,5,200)';
    curid=1;
    % Generate an id for each detection we process, so that we can keep
    % track of the associated data later.
    for(i=1:numel(evaldets))
      ds.evaldetids{i}=c(curid:curid+size(ds.evaldets{i},1)-1);
      curid=curid+size(ds.evaldets{i},1);
    end
    ds.round1.npredits=1;
    ds.round1.torun=1:numel(evaldets);
    torun=ds.round1.torun;
    ds.conf.checksaves=1;
  end
  roundid=1;
  while(dsfield(ds,['round' num2str(roundid)],'finishedverify')),roundid=roundid+1;end
  while(roundid<=5)
    rnd=num2str(roundid);
    nextrnd=num2str(roundid+1);
    dsdelete(['ds.round' nextrnd '.*']);
    dsrundistributed(['try,'...
                   'stuffmodelgmm=dsload(''ds.stuffmodelgmm'');'...
                   'conf=ds.conf;'...
                   'evalidx=ds.round' rnd '.torun(dsidx);'...
                   ... % load the patch detections associated with this cluster
                   'mydets=dsload([''ds.evaldets{'' num2str(evalidx) ''}''],''clear'');'...
                   ... % the 'state' keeps track of all internal data generated by verifyelementdist. On round 1
                   ... % it's just [] since there's no state, but afterward it contains things like the 
                   ... % probability heatmaps for each image, verification scores, which patches have been run, etc.
                   'if(' rnd '==1),'...
                      'state=[];'...
                   'else,'...
                     'state=dsload([''ds.round' rnd '.states{'' num2str(evalidx) ''}''],''clear'');'...
                   'end,'...
                   'npreds=dsload(''ds.round' rnd '.npredits'');'...
                   '[ds.round' nextrnd '.scores{evalidx},state]=verifyelementdist(mydets,stuffmodelgmm,state,npreds,conf);'...
                   'ds.round' nextrnd '.states{evalidx}=state;'...
                   ... % Finally, generate a mask of where the algorithm thinks the object is in the image.  Only
                   ... % the masks generated on round 5 are used; they're ultimately used to improve deduplication
                   ... % at the end.
                   'for(i=1:numel(state.preds)),'...
                     'pr=state.preds{i};'...
                     'if(isempty(pr)),continue;end,'...
                     'hms{i}=pr.hm;'...
                     'bghms{i}=pr.bghm;'...
                     'predicteds{i}=pr.predictednooc;'...
                     'inimpyrlev(i)=pr.inimpyrlevel;'...
                     'conf2.thresh(i)=state.thresh(i);'...
                   'end,'...
                   'conf2.returnmask=true;'...
                   'masks=c(displaylikelihoodmask(mydets,hms,bghms,predicteds,inimpyrlev,conf2));'...
                   'if(numel(masks)<size(mydets,1)),masks{size(mydets,1),1}=[];end,'...
                   'for(i=1:numel(masks)),'...
                     'if(~isempty(masks{i})),'...
                       ... % if the detection was in a flipped image, flip the mask
                       'if(mydets(i,8)),' ...
                         'masks{i}=masks{i}(:,end:-1:1);'...
                       'end,'...
                       'masks{i}=struct(''mask'',masks{i},''inimpyrlevel'',state.preds{i}.inimpyrlevel,''imsize'',size(getimg(mydets(i,7))));'...
                     'end,'...
                   'end,'...
                   'ds.round' nextrnd '.masks{evalidx}=masks;catch ex,dsprinterr;end'...
                   ],['ds.round' rnd '.torun'],struct('noloadresults',true));

    % The remainder of the loop just figures out which elements to retain for the next round.
    % See section 4.1 of the paper.
    % we greedily maximize the objective \sum_{i,j}2^{-j}s^{\chi}_{i,j}, where \chi is the set of
    % elements selected, and s^{\chi}_{i,j} is the score of the j'th highest-scoring
    % patch in image i out of all detections selected in \chi.

    % First we aggregate all the scores, and create a 3-column index of these scores
    % in the form [detector_id, image_id, score]
    torun=dsload(['ds.round' rnd '.torun']);
    npredits=dsload(['ds.round' rnd '.npredits']);
    dsup(['ds.round' nextrnd '.npredits'],npredits*2);
    predictions={};
    allscores=dsload(['ds.round' nextrnd '.scores'],'clear');
    for(i=numel(allscores):-1:1)
      if(~isempty(allscores{i}))
        scrs=allscores{i};
        scrs(isnan(scrs))=-Inf;
        [~,idx]=maxk(scrs,npredits-1);
        scrs(scrs<0)=0;
        predictions{i,1}=[ds.evaldets{i}(idx,6:7),scrs(idx)];
      else
        predictions{i,1}=zeros(0,3);
      end
    end
    predictionsbyim=cell2mat(predictions);

    % compute the contribution of each detector to the total
    % objective if it gets added to the active set.
    contribution=cellfun(@(x) nansum(x(:,3)),predictions);
    % add an extra column to the index of scores; this will keep track of how
    % much each detection is contributing to the objective while accounting for halving.
    predictionsbyim=[predictionsbyim predictionsbyim(:,3)];
    % distribute by image.  In the final array, make sure we can index by image id.
    [predictionsbyim,imid]=distributeby(predictionsbyim,predictionsbyim(:,2));
    predictions2=cell(max(imid),1);
    predictions2(imid)=predictionsbyim;
    predictionsbyim=predictions2;

    % how many to select
    torunnew=zeros(max(1000,round(numel(torun)/2)),1);
    fincontrib=zeros(size(torunnew));
    nselected=0;
    while(nselected<numel(torunnew))
      % select the one that contributes most to the objective.
      [~,selection]=max(contribution);
      fincontrib(nselected+1)=contribution(selection);
      contribution(selection)=-Inf;
      % grab the detections for the selected detector
      preds=predictions{selection};
      % for each detector, go to its image and halve the contribution
      % of anything that scored less than that.  Note that we measure
      % which things scored less by the third column (the one that doesn't
      % get halved) and actually halve the fourth one.  For each score
      % that gets halved, update the contribution of that detector.
      for(i=1:size(preds,1))
        imid=preds(i,2);
        score=preds(i,3);
        pbi=predictionsbyim{imid};
        idx=pbi(:,3)<score;
        pbi(idx,4)=pbi(idx,4)/2;
        % note there is a max of one detection per image; otherwise this
        % code may assign the same index twice...
        contribution(pbi(idx,1))=contribution(pbi(idx,1))-pbi(idx,4);
        predictionsbyim{imid}=pbi;
      end
      nselected=nselected+1;
      torunnew(nselected)=selection;
    end
    dsup(['ds.round' nextrnd '.torun'],sort(torunnew));
    dsup(['ds.round' rnd '.finishedverify'],true);
    dssave;
    dsclear(['ds.round' rnd])
    roundid=roundid+1;
    rnd=num2str(roundid);
  end
end

dsload('ds.finishedmaskdisplay')
if(~dsbool(ds,'finishedmaskdisplay'))
  dsdelete('ds.verified_display');
  dsdelete('ds.unverified_display');
  dsdelete('ds.dedupeord');
  dsload('ds.round*.finishedverify')
  roundid=1;
  while(dsfield(ds,['round' num2str(roundid)],'finishedverify')),roundid=roundid+1;end
  dsload(['ds.round' num2str(roundid-1) '.torun']);
  % generate the patch with the fuzzy mask for each verified detection.
  % This masked patch will appear in the final display of the discovered clusters.
  dsrundistributed([...
                   'evalidx=ds.round' num2str(roundid-1) '.torun(dsidx);'...
                   'mydets=dsload([''ds.evaldets{'' num2str(evalidx) ''}''],''clear'');'...
                   'myids=dsload([''ds.evaldetids{'' num2str(evalidx) ''}''],''clear'');'...
                   'state=dsload([''ds.round' num2str(roundid) '.states{'' num2str(evalidx) ''}''],''clear'');'...
                   'scores=dsload([''ds.round' num2str(roundid) '.scores{'' num2str(evalidx) ''}''],''clear'');'...
                   'scores(isnan(scores))=-Inf;'...
                   '[~,ord]=sort(scores,''descend'');'...
                   'ord(scores(ord)<-1000000)=[];'...
                   'for(i=1:numel(ord)),'...
                     'pr=state.preds{ord(i)};'...
                     'hms{i}=pr.hm;'...
                     'bghms{i}=pr.bghm;'...
                     'predicteds{i}=pr.predictednooc;'...
                     'inimpyrlev(i)=pr.inimpyrlevel;'...
                   'end,'...
                   'immask=displaylikelihoodmask(mydets(ord,:),hms,bghms,predicteds,inimpyrlev);'...
                   'for(i=1:numel(ord)),'...
                     'ds.verified_display.patchimg{myids(ord(i))}=immask{i};'...
                   'end,'...
                   ],['ds.round' num2str(roundid-1) '.torun'],struct('noloadresults',true));
  dsload('ds.evaldet*s');
  evaldets=ds.evaldets;
  detrscore=[];
  findetrs=dsload(['ds.round' num2str(roundid-1) '.torun']);
  % compute a score for each detector by summing the scores of each verified patch.
  for(i=1:numel(ds.evaldets))
    if(ismember(i,findetrs))
      scores=dsload(['ds.round' num2str(roundid) '.scores{' num2str(i) '}']);
      scores(isnan(scores))=-Inf;
      detrscore(idxof(i,findetrs),1)=sum(scores(scores>0));
      % don't display the seed patch; its mask tends to look weird since it gets predicted first, and
      % at that point the thing model was doing a bad job predicting where it could predict.
      evaldets{i}(:,5)=scores;
    end
  end
  m=dsload(['ds.round' num2str(roundid) '.masks'])';
  masks=cat(1,m{:});
  clear m;
  detrids=dsload(['ds.round' num2str(roundid-1) '.torun']);
  inds=find(ismember(1:numel(ds.evaldetids),detrids));
  ovldets=cell2mat(dsload(['ds.evaldets{' num2str(inds(:)') '}'])');
  if(size(masks,1)~=size(ovldets,1))
    disp('mask ovldet mismatch');
    keyboard;
  end

  % Now that the order is established, perform greedy deduplication
  idxs=findOverlapping4(ovldets,masks,[],[],struct('sortscores',[detrids(:) detrscore(:)],'maxoverlaps',12,'overlapthresh',.2));
  mhprender('patchdisplay.mhp','ds.verified_display.deduphtml',struct('detrord',idxs,'dets',cell2mat(evaldets(:)),'maxperdet',20,'patchwidth',150,'ctrbox',[50 50 100 100]));
  ds.dedupeord=idxs;

  % Now display the clusters in the same order, but this time show the top patches retrieved by LDA,
  % so we can see how much the verification procedure improved things.
  evd=maxkall(ds.evaldets(idxs),5,21);
  for(i=1:numel(evd)),evd{i}(1,:)=[];end
  evd=cell2mat(evd(:));
  ds.unverified_display.patchimg=extractpatches(evd);
  mhprender('patchdisplay.mhp','ds.unverified_display.deduphtml',struct('detrord',idxs,'dets',evd,'maxperdet',20));
  dssave;
end
% Finally, run the car keypoint experiment.
nm=ds.imgs{ds.conf.currimset}.fullname;
for(i=1:numel(nm))
  pos=find(nm{i}=='/');
  if(~isempty(pos))
    nm{i}=nm{i}(pos(end)+1:end);
  end
end
imids=cellfun(@(x) x(1:11),nm,'UniformOutput',false);
load('2011_PASCAL_Car_Landmark');
ds.annotimidx=idxof({annotation.image},imids);
ds.annot=annotation;

% This file contains the list of images containing at 
% least one car larger than 150 pixels on its smallest side.
f=fopen('bigcars.txt');
bigcars=textscan(f,'%s');
fclose(f);
ds.runimgs=idxof(bigcars{1},nm);

% extract the detections from just those images
allevaldets=cell2mat(c(dsload('ds.evaldets')));
dsload('ds.round*.finishedverify');
roundid=1;
while(dsfield(ds,['round' num2str(roundid)],'finishedverify')),roundid=roundid+1;end
allevaldets=allevaldets(ismember(allevaldets(:,6),dsload(['ds.round' num2str(roundid-1) '.torun'])),:);
[imdets,detimid]=distributeby(allevaldets,allevaldets(:,7));
imdets(~ismember(detimid,ds.runimgs))=[];
detimid(~ismember(detimid,ds.runimgs))=[];
ds.kppreddets(idxof(detimid,ds.runimgs))=imdets;
dsrundistributed([...
                  'dsload(''ds.stuffmodelgmm'');dsload(''ds.annot'');dsload(''ds.annotimidx'');'...
                  'ds.kps{dsidx}=predictkeypoints(ds.kppreddets{dsidx},''ds.evaldets'',''ds.round' num2str(roundid-1) '.states'',ds.annot,ds.annotimidx,ds.stuffmodelgmm,ds.conf);'...
                 ],'ds.kppreddets');
[ds.prec,ds.rec]=evalkeypoints(ds.kps,ds.runimgs,ds.annot,ds.annotimidx);


%If you want to look at the keypoints you predicted, now is the time.
if(0)
  dispkeypoints;
end
