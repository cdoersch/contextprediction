% Written by Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% Given a set of image id's (whose indices are given in ds.myiminds(dsidx)),
% extract random patches from all of them and save the HOG features in 
% ds.pats{dsidx} (and the corresponding image id's in ds.imid{dsidx}).  
% The dictionary will overall have about 1,000,000 patches.
%
% Internally, this works very much like a streaming selection algorithm:
% we sample random patches and store them in lib, randomly replacing
% old patches when we run out of room.

conf=ds.conf;
if(~isfield(conf,'n'))
  %conf.n=1000000;
  conf.n=ceil(1000000/numel(ds.myiminds)+1.05)*numel(dsidx);
end
if(~isfield(conf,'patchsz'))
  conf.patchsz=7;
end
try
lib=zeros(1,conf.n,'single');
libimid=zeros(conf.n,9);
nextidx=1;
myiminds=ds.myiminds;
for(imidx=dsidx(:)')
  disp(['running im ' num2str(imidx)]);
  im=getimg(myiminds(imidx));
  bbs=[1 1 size(im,2) size(im,1) 0];
  %bbs=scaledets(bbs,1.5);
  pats=extractpatches([bbs repmat([0,myiminds(imidx)],size(bbs,1),1) ],[],struct('noresize',true));
  for(bbidx=1:size(bbs,1))
    scalefact=1;%sqrt(300*100/prod(size(pats{bbidx}(:,:,1))));
    pats{bbidx}=imresize(im2double(pats{bbidx}),scalefact);
    pyramid=constructFeaturePyramidForImg(pats{bbidx},ds.conf.params);
    pcs=[conf.patchsz conf.patchsz];
    if(numel(pyramid.features)==0)
      continue;
    end
    

    pcs(3)=size(pyramid.features{1},3);
    pcs(4)=0;
    [features, levels, indexes,gradsums] = unentanglePyramid(pyramid, ...
      pcs,struct('whitening',false,'normbeforewhit',false,'normalizefeats',false));
    invalid=(gradsums<9);
    features(invalid,:)=[];
    levels(invalid)=[];
    indexes(invalid,:)=[];
    gradsums(invalid)=[];
    patsz=ds.conf.params.patchCanonicalSize;%allsz(resinds(k),:);
    fsz=(patsz-2*ds.conf.params.sBins)/ds.conf.params.sBins;
    pos=pyridx2pos(indexes,reshape(levels,[],1),fsz,pyramid);
    pos=[pos.x1 pos.y1 pos.x2 pos.y2];
    pos=(pos-.5)/scalefact+.5;
    pos=bsxfun(@plus,pos,[bbs(bbidx,1) bbs(bbidx,2) bbs(bbidx,1) bbs(bbidx,2)]);

    if(nextidx<size(lib,2))
      inds=nextidx:min(size(lib,2),nextidx-1+size(features,1));
      finds=inds-nextidx+1;
      if(size(lib,1)==1)
        lib(size(features,2),1)=0;
      end
      lib(:,inds)=single(features(finds,:))';
      features(finds,:)=[];
      nextidx=nextidx+numel(finds);
      libimid(inds,:)=[pos(finds,:) repmat([0 0 myiminds(imidx),0,bbs(bbidx,5)],numel(inds),1)];
      pos(finds,:)=[];
    end
    if(~isempty(features))
      rp=randperm(size(features,1));
      ntokeep=round(size(features,1)*(size(lib,2)/(size(features,1)+nextidx)));
      sel=zeros(size(features,1),1);
      sel(rp(1:ntokeep))=1;
      nextidx=nextidx+size(features,1);
      features(~sel,:)=[];
      pos(~sel,:)=[];
      rp=randperm(size(lib,2));
      lib(:,rp(1:size(features,1)))=single(features)';
      libimid(rp(1:size(features,1)),:)=[pos repmat([0 0 myiminds(imidx),0,bbs(bbidx,5)],size(features,1),1)];
    end
  end
end
valid=~all(lib==0,1);
ds.pats{dsidx}=lib(:,valid);
ds.imid{dsidx}=libimid(valid,:);
ds.n{dsidx}=sum(valid);

catch ex,dsprinterr;end
