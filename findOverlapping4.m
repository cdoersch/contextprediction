% datasource: a matrix of detections, or a path in dswork containing
% detections, in which case the detections will be read one column at
% a time, and it is assumed that all detections for one detector will
% be contained in a single cell of datasource.  in this case, inds will
% contain the indices to read; otherwise it is ignored.
% masks: a cell array parallel with datasource, which can specify a probability
% mask over a HOG grid over the image (or it can be []).  These will be 
% used to compute better overlap scores.  For a pair of detections in the 
% same image, if both have an available mask, then we compute overlap as
% sum(mask1(:).*mask2(:))/sqrt(sum(mask1(:).^2)*sum(mask2(:).^2)).  If one
% mask is available but not the other, we compute overlap by averaging the mask
% within the bounding box.
% classfordetr: a two-column matrix where the first column is a detector id,
% and the second is the class that the detector was sampled from.
% conf can contain:
%   'overlapthresh': the threshold for two patches in the same image to be considered
%                    overlapping
%   'ndetsforoverlap': maximum number of detections in common before two detectors are
%                      clustered together
%   'clusterer': 'greedy' implies the greedy de-duplication procedure described on
%                the SIGGRAPH '12 paper.  'agglomerative' is the agglomerative clustering
%                algorithm used to cluster patches in the NIPS '13 paper.
 
%detids: ids of nonoverlapping detectors, sorted by score
%scores: scores of detectors in detids, synchronized with detids
%component: the component of overlapping detectors; first column is all detids sorted by scores, 
%    second is the root detid that a given detector overlaps with, 
%    third is score corresponding to detector
function [detids,scores,component]=findOverlapping4(datasource,masks,inds,classfordetr,conf)
  %masks=[cell(size(datasource,1),1);masks]
  %datasource=[datasource;detformasks];
  if(~exist('conf','var'))
    conf=struct();
  end
  if(~dsfield(conf,'overlapthresh'))
    conf.overlapthresh=.3;
  end
  if(~dsfield(conf,'maxoverlaps'))
    conf.maxoverlaps=5;
  end
  if(~dsfield(conf,'ndetsforoverlap'))
    conf.ndetsforoverlap=Inf;
  end
  if(~dsfield(conf,'clusterer'))
    conf.clusterer='greedy';
  end
  imgs=dsload('.ds.imgs{ds.conf.currimset}');
  try
  global ds;
  alldata={};
  scores=[];
  detids=[];
  if(ischar(datasource))
    disp('loading...')
    for(i=inds(:)')
      dat=structcell2mat(dsload([datasource '{1:' num2str(dssavestatesize(datasource,1)) '}{' num2str(i) '}'],'clear'));
      if(isstruct(dat))
        dat=det2mat(dat);
      end
      if(isempty(dat))
        dat=zeros(0,7);
      end
      [~,posinclassfordetr]=ismember(dat(:,6),classfordetr(:,1));
      positivedets=dat(classfordetr(posinclassfordetr,2)==imgs.label(dat(:,7)),:);
      positivedets=distributeby(positivedets,positivedets(:,6));
      for(k=1:numel(positivedets))
        ndfo=conf.ndetsforoverlap;
        if(ndfo>0&ndfo<1)
           ndfo=round(sum(positivedets{k}(:,5)>0)*ndfo);
        end
        [~,tmpinds]=maxk(positivedets{k}(:,5),ndfo);
        alldata{end+1,1}=positivedets{k}(tmpinds,[1:4,6:7]);
      end
      clear positivedets;
      mydetids=unique(dat(:,6));
      for(j=1:numel(mydetids))
        if(~isfield(conf,'sortscores'))
          %if(sortcount)
            %ispos=classfordetr(posinclassfordetr,2)==imgs.label(dat(:,7));
            %ip=ispos(dat(dat(:,6)==mydetids(j),7));
            %scrtmp=dat(dat(:,6)==mydetids(j),5);
            %[~,ord]=sort(scrtmp,'descend');
            %if(dsfield(conf,'sortn'))
            %  sortn=conf.sortn;
            %else
            %  sortn=30;
            %end
            scores(end+1)=0;%sum(ip(ord(1:conf.sortn)));
          %else
          %  scores(end+1)=scoreFromDots(dat(dat(:,6)==mydetids(j),5),ispos(dat(dat(:,6)==mydetids(j),7)+1),ds.conf.params);
          %end
        end
        detids(end+1)=mydetids(j);
      end
      disp(i);
    end
    if(dsfield(conf,'sortscores'))
      [~,ord]=ismember(detids,conf.sortscores(:,1));
      scores=conf.sortscores(ord,2);
    end
    alldata=cell2mat(alldata);
  else
    if(~(dsfield(conf,'sortn') || dsfield(conf,'sortscores')))%stupid backwards compatibility
      detids=sort(unique(datasource(:,6)));
      for(i=numel(detids):-1:1)
        inds=find(datasource(:,6)==detids(i));
        [~,~,counter]=unique(imgs.label(datasource(inds,7)));
        [scores(i),maxid]=max(histc(counter,1:max(counter)));
        scores(i)=scores(i)/numel(counter);
        alldata{i}=datasource(inds(counter==maxid),[1:4,6:7]);
      end
    else
      [datasource,masks,detids]=distributeby(datasource,masks,datasource(:,6));
      if(dsfield(conf,'sortscores'))
        [~,ord]=ismember(detids,conf.sortscores(:,1));
        scores=conf.sortscores(ord,2);
        for(i=numel(datasource):-1:1)
          [~,tmpinds]=maxk(datasource{i}(:,5),conf.ndetsforoverlap);
          alldata{i}=datasource{i}(tmpinds,[1:4,6:7]);
          allmasks{i}=masks{i}(tmpinds);
        end
      else
        for(i=numel(datasource):-1:1)
          [~,posinclassfordetr]=ismember(datasource{i}(:,6),classfordetr(:,1));
          isdetpos=classfordetr(posinclassfordetr,2)==imgs.label(datasource{i}(:,7));
          [~,detord]=maxk(datasource{i}(:,5),conf.sortn);
          scores(i)=sum(isdetpos(detord))/conf.sortn;
          posdets=datasource{i}(isdetpos,:);
          ndfo=conf.ndetsforoverlap;
          if(ndfo>0&ndfo<1)
             ndfo=round(size(posdets,1)*ndfo);
          end
          [~,tmpinds]=maxk(posdets(:,5),ndfo);
          alldata{i}=posdets(tmpinds,[1:4,6:7]);

        end
      end
    end
    %alldata=datasource(:,[1:4,6:7]);
    alldata=cell2mat(alldata');
    allmasks=cat(1,allmasks{:});
    clear datasource;
  end
  disp('sort by image');
  [~,ord]=sort(scores,'descend');
  [~,alldata(:,5)]=ismember(alldata(:,5),detids(ord));%relabel detector id's such that highest scoring detectors get lowest id's
  %[~,imgord]=sort(alldata(:,6));
  %alldata=alldata(imgord,:);
  %alldata=mat2cell(alldata,diff([0;find(diff(alldata(:,6))); size(alldata,1)]),size(alldata,2));%break up alldata into a cell array by image
  [alldata,allmasks]=distributeby(alldata,allmasks,alldata(:,6));
  disp('compute overlaps');
  overlapping={};
  for(i=1:numel(alldata))
    overl = computeOverlap(alldata{i}(:,1:4), alldata{i}(:,1:4), 'pascal');
    masks=find(~cellfun(@isempty,allmasks{i}));
    %if(alldata{i}(1,6)==4724)
    %  keyboard
    %end
    for(m1=1:numel(masks))
      for(k=1:numel(masks))
        if(masks(m1)==k),continue;end
        if(ismember(k,masks))
          mask1=padarray(allmasks{i}{masks(m1)}.mask,[1,1]);
          mask2=padarray(allmasks{i}{k}.mask,[1,1]);
          maxsz=max(size(mask1),size(mask2));
          mask1=imresize(mask1,maxsz);
          mask2=imresize(mask2,maxsz);
          ovl=sum(min(mask1(:),mask2(:)))/sum(max(mask1(:),mask2(:)));
        else
          imsz=allmasks{i}{masks(m1)}.imsize(1:2);%,1:2);
          inimpyrlevel=allmasks{i}{masks(m1)}.inimpyrlevel;%alldata{i}(masks(m1),3);
          reszbox=((alldata{i}(k,[1:4])-.5).*repmat(round(imsz([2 1])*2^(-(inimpyrlevel-1-8)/8))./imsz([2 1]),[1 2])+.5)/ds.conf.params.sBins-1;
          reszbox(1:2)=max(1,floor(reszbox(1:2)));
          reszbox([4 3])=min(size(allmasks{i}{masks(m1)}.mask),ceil(reszbox([4 3])));
          ovl=sum(sum(allmasks{i}{masks(m1)}.mask(reszbox(2):reszbox(4),reszbox(1):reszbox(3))))/((reszbox(4)-reszbox(2)+1)*(reszbox(3)-reszbox(1)+1));
        end
        overl(masks(m1),k)=ovl;
        overl(k,masks(m1))=ovl;

      end
    end
    overl(1:(size(overl,1)+1):end)=0;
    [r,col]=find(overl>conf.overlapthresh);
    ovldets=[alldata{i}(r,5) alldata{i}(col,5)];
    ovldets=ovldets(ovldets(:,1)<ovldets(:,2),:);
    overlapping{i,1}=ovldets;
    if(mod(i,100)==0)
      disp(i);
    end
  end
  clear alldata;
  overlapping=sortrows(cell2mat(overlapping));
  hasdiff=diff(overlapping(:,1))+diff(overlapping(:,2));
  ends=[find(hasdiff); size(overlapping,1)];
  hists=diff([0;ends]);
  if(strcmp(conf.clusterer,'greedy'))
    links=overlapping(ends(hists>conf.maxoverlaps),:);
    links=sparse(links(:,2),links(:,1),ones(size(links(:,1))),numel(detids),numel(detids));
    hasoverlap=zeros(numel(detids),1);
    hasoverlap(unique(overlapping(ends(hists>10),2)))=1;
    old=ones(size(hasoverlap(:)));
    while(~all(hasoverlap==old))
      old=hasoverlap;
      hasoverlap=links*(~hasoverlap)>0;
    end

    eliminate=find(hasoverlap);

    if(nargout>=3)
      component=zeros(numel(hasoverlap),2);
      component(:,1)=detids(ord);
      for(i=1:size(component,1))
        if(hasoverlap(i))
          mylinks=links(i,(1:(i-1))).*(~hasoverlap(1:(i-1)))';
          mylinks=find(mylinks);
          component(i,2)=component(mylinks(1),2);
        else
          component(i,2)=component(i,1);
        end
      end
    end
  elseif(strcmp(conf.clusterer,'agglomerative'))
    links=overlapping(ends(hists>0),:);
    links=sparse(links(:,2),links(:,1),hists(hists>0),numel(detids),numel(detids));
    links=max(links,links');
    minval=max(links(:));
    cluststr=linkage2((1:numel(detids))','average',{@(x,y) -links(x,y)+minval});
    component=cluster(cluststr,'cutoff',minval-conf.maxoverlaps+.001,'criterion','distance');
    component2=component;
    %detrrank(ord)=1:numel(ord);
    for(j=unique(component(:)'))
      mindetrrank=min(find(component==j));
      component2(component==j)=detids(ord(mindetrrank));
    end
    component=[c(detids(ord)) component2(:)];
    eliminate=component(:,1)~=component(:,2);
  else
    error('unrecognized clusterer');
  end
  scores=scores(:);
  if(nargout>=3)
    component=[component scores(ord)];
  end
  detids=detids(ord);%(ord(eliminate))=[];
  detids(eliminate)=[];
  scores=scores(ord);%(ord(eliminate))=[];
  scores(eliminate)=[];
  catch ex,dsprinterr;end
end
