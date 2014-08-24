% Written by Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% Given a set of element bounding boxes in dets, all from the same image,
% use the other detections it corresponds to to transfer keypoints from other
% images to the current image.  Uses contextpredict to estimate corresp
% as well as a per-HOG-cell confidences.
% Then it uses correspwarp to estimate a pixel-to-pixel warping to transfer
% keypoints.  Finally it performs de-duplication of keypoints over regions
% that it estimates to be the same object instance.
%
% dets: a set of bounding boxes in the standard matrix form.
%
% detsbydetrvar: the variable containing the set of detectons organized by detector.
%                This is how we find the predictor images.  This variable will be a
%                cell array that can be indexed by detector id.
%
% statesvar: the set of states estimated from verifyelementcorresp.  This is used to
%            (a) get a probability mask for each predictor image, and (b) order the
%            predictor images by the likelihood that they contain the object of interest
%
% keypoints/keypointimgs: parallel arrays specifying ground truth keypoints.  keypoints
%                         is a struct array read from 2011_PASCAL_Car_Landmark.mat, and
%                         keypointimgs specifies the image id's associated with each element
%                         of that array.
%
% stuffmodelgmm: the stuff model
%
% res: the predicted keypoints, in the format [x y keypoint_type score].
% 
function res=predictkeypoints(dets,detsbydetrvar,statesvar,keypoints,keypointimgs,stuffmodelgmm,conf)
try
  global ds;
  if(~exist('conf','var'))
    conf=struct();
  end
  % If we flip the image, left-right labels for all the keypoints need to be switched
  lrperm=[2 1 14 5 4 7 6 9 8 11 10 13 12 3];
  if(isempty(dets))
    res=[];
    return;
  end
  inim=getimg(dets(1,[7:8]));
  for(i=1:size(dets,1))
    % First estimate correspondence using 20 predictor images for one of the detections.
    state=dsload([statesvar '{' num2str(dets(i,6)) '}'],'clear');
    othdets=dsload([detsbydetrvar '{' num2str(dets(i,6)) '}'],'clear');
    prevpreds=state.probs;
    prevpreds(isnan(prevpreds))=-Inf;
    % Get rid of other detections from the same image
    prevpreds(othdets(:,7)==dets(i,7))=-Inf;
    [~,ord]=sort(prevpreds,'descend');
    ord(prevpreds(ord)==-Inf)=[];
    ord=ord(1:min(numel(ord),20));
    [hm,bghm,corresp,pyrs,inimpyrlevel,blurconf,usemap,predicted,have_preds_orig,predictednooc]=contextpredict(dets(i,:),othdets(ord,:),stuffmodelgmm,state.certaintymap(ord),conf);
    disp(['predictkeys ' num2str(i) '/' num2str(size(dets,1))]);
    % If the confidence is low, drop this detection.
    if(sum(sum((hm-bghm).*(predicted>.3)))<100),continue;end
    totscr=sum(sum((hm-bghm).*(predicted>.3)));
    transfkp=cell(14,1);
    confmap=blurconf.*predictednooc;
    confmap(isnan(confmap))=0;
    % For each keypoint in the predictor images, estimate the confidence with which
    % we would transfer it.  This includes the confidence score of the target location
    % in the query image, the total confidence that the query image contains the
    % thing, and the usage score.
    for(j=1:numel(ord))
      if(ismember(othdets(ord(j),7),keypointimgs))
        im=getimg(othdets(ord(j),[7:8]));
        [~,warpx,warpy,pixusage]=correspwarp(corresp,ds.conf.params.sBins,im,pyrs,j,struct('nowarp',true));
        [idxall]=find(othdets(ord(j),7)==keypointimgs);
        for(kpidx=idxall(:)')
          kp=keypoints(kpidx).keypoints;
          for(k=1:size(kp,1))
            if(kp(k,1)>0&&kp(k,2)>0)
              rndkp=round(kp(k,1:2));
              tostorek=k;
              if(othdets(ord(j),8))
                rndkp(1)=-rndkp(1)+size(im,2)+1;
                tostorek=lrperm(k);
              end
              rndkp=max(rndkp,1);
              rndkp(1)=min(rndkp(1),size(pixusage,2));
              rndkp(2)=min(rndkp(2),size(pixusage,1));
              if(pixusage(rndkp(2),rndkp(1))<1e-8)
                continue;
              end
              warpkp=[warpx(rndkp(2),rndkp(1)),warpy(rndkp(2),rndkp(1))];
              if(any(isnan(warpkp)))
                continue;
              end
              imsc=size(im(:,:,1))./round(size(im(:,:,1)).*2^(-(inimpyrlevel-1-8)/8));
              idx=round(warpkp/ds.conf.params.sBins)-1;
              idx=idx([2 1]);
              if(idx(1)>size(confmap,1)+1)
                error('something didnt scale');
              end
              idx=max(1,min(idx,size(confmap)));
              warpkp=warpkp.*imsc([2 1]);
              % k=keypoint type, kpidx=the image where the keypoint came from, othdets=whether or not this keypoint came from a flipped image. tostorek is the
              % keypoint type after correcting for the flip of the predictor image, but not correcting for the flip of the query image.
              transfkp{tostorek}(end+1,1:8)=[warpkp,tostorek,confmap(idx(1),idx(2))*totscr*min(1,usemap(idx(1),idx(2),j)/max(usemap(idx(1),idx(2),:))),confmap(idx(1),idx(2)).*usemap(idx(1),idx(2),j),k,kpidx,othdets(ord(j),8)];
            end
          end
        end
      end
    end
    for(k=1:numel(transfkp))
      if(~isempty(transfkp{k}))
        [~,idx]=max(transfkp{k}(:,5));
        transfkp{k}=transfkp{k}(idx,:);
      end
    end
    alltransfkp{i}=cell2mat(c(transfkp));
    rszmask=imresize(confmap,8*2^((inimpyrlevel-1-8)/8));
    allmasks{i}=zeros(size(inim(:,:,1)));
    allmasks{i}(1:size(rszmask,1),1:size(rszmask,2))=rszmask;
    allmasks{i}=allmasks{i}./sum(sum(allmasks{i}));
    % now we correct for the flip of the query image.
    if(dets(i,8))
      allmasks{i}=allmasks{i}(:,end:-1:1);
      if(~isempty(alltransfkp{i}))
        alltransfkp{i}(:,1)=-alltransfkp{i}(:,1)+size(inim,2)+1;
        alltransfkp{i}(:,3)=lrperm(alltransfkp{i}(:,3));
        alltransfkp{i}(:,8)=~alltransfkp{i}(:,8);
      end
    end
  end
  if(~exist('allmasks','var'))
    res=[];
    return;
  end
  bad=cellfun(@isempty,allmasks);
  allmasks(bad)=[];
  alltransfkp(bad)=[];
  % finally, cluster the set of masks into objects, and deduplicate within each object.
  for(i=1:numel(allmasks))
    for(j=1:numel(allmasks))
      ovl(i,j)=sum(sum(min(allmasks{i},allmasks{j})))./sum(sum(max(allmasks{i},allmasks{j})));
      ovl(j,i)=ovl(i,j);
    end
  end
  ovl=sparse(ovl>.2);
  [~,comps]=graphconncomp(ovl,'Directed',false);
  outkp={};
  for(i=unique(comps))
    kp=cell2mat(c(alltransfkp(find(comps==i))));
    if(isempty(kp))
      continue;
    end
    kp=distributeby(kp,kp(:,3));
    kp=maxkall(kp,4,1);
    outkp{end+1,1}=cell2mat(kp);
  end
  res=cell2mat(outkp);
catch ex,dsprinterr;edn
end
