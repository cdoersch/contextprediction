% Given predicted keypoints and groundtruth keypoints, compute a precision
% recall curve.  A keypoint is considered to be correct if it's within 10%
% of the max dimension of the bounding box associated with the groundtruth keypoint.
%
% predkeys/imid's: predicted keypoints distributed by image.  For the i'th image,
%                  imids(i) is an index into ds.imgs{ds.conf.currimset} (although 
%                  this is never used; it just needs to let us match the predictions
%                  with keyimids).  predkeys{i} then contains the predicted keypoints as
%                  a n-by-4 matrix with columns [x y keypoint_type score].
%
% gtkeypoints/keyimids: keyimids(i) is likewise an index into ds.imgs{ds.conf.currimset}
%                       saying where each keypoint came from.  The corresponding
%                       gtkeypoints(i) struct gives the ground truth keypoints for one
%                       bounding box (note keyimids may have duplicates; imids will not).
%                       gtkeypoints is in the format of 2011_PASCAL_Car_Landmarks.mat.
%                       (from https://github.com/mhejrati/3D-Object/tree/master/data).
function [prec,rec]=evalkeypoints(predkeys,imids,gtkeypoints,keyimids)
global ds;
try
  correcterrscore=cell(size(imids));
  %bbs=ds.bboxes{ds.conf.currimset};
  %bbs=effstridx(bbs,bbs.label==7);
  %[bbs,bbimids]=distributeby(bbs,bbs.imidx);
  %bbs=idxwithdefault(bbs,idxof(imids,bbimids),{});
  ngtkeys=0;
  for(i=1:numel(gtkeypoints))
    if(~ismember(keyimids(i),imids))
      continue;
    end
    bb=gtkeypoints(i).bbox;
    %if(bb(3)-bb(1)+1<150||bb(4)-bb(2)<150)
    %  continue
    %end
    ngtkeys=ngtkeys+sum(gtkeypoints(i).keypoints(:,1)~=0|gtkeypoints(i).keypoints(:,2)~=0);
  end
  [gtkeypoints,keyimids]=distributeby(c(gtkeypoints),c(keyimids));
  %keyboard
  gtkeypoints=idxwithdefault(gtkeypoints,idxof(imids,keyimids),{[]});
  %keyboard
  for(i=1:numel(predkeys))%numel(imids))
    if(~isempty(predkeys{i}))
      correcterrscore{i}=zeros(size(predkeys{i},1),3);
      correcterrscore{i}(:,3)=predkeys{i}(:,4);
      [~,ord]=sort(predkeys{i}(:,4),'descend');
      used=cell(numel(gtkeypoints{i}),1);
      %keyboard
      for(pk=ord(:)')
        inannotatedbb=false;
        dist=Inf;
        mykey=predkeys{i}(pk,:);
        for(gk=1:numel(gtkeypoints{i}))
          if(isempty(used{gk})),used{gk}=zeros(size(gtkeypoints{i}(gk).keypoints,1),1);end
          %for(kpidx=1:numel(gtkeypoints{i}(gk).keypoints(:,1)))
          %  gtkp=gtkeypoints{i}(gk).keypoints(kpidx,1:2);
            gtkp=gtkeypoints{i}(gk).keypoints(predkeys{i}(pk,3),1:2);
            if(gtkp(1)==0&&gtkp(2)==0),continue;end
            bb=gtkeypoints{i}(gk).bbox;
            newdist=sqrt(sum((gtkp-predkeys{i}(pk,1:2)).^2))/max((bb(3)-bb(1)+1),(bb(4)-bb(2)+1));
            if(newdist<dist)
              usedidx=gk;
              dist=newdist;
            end
            if(mykey(1)>bb(1)-.5&&mykey(1)<bb(3)+.5&&mykey(2)>bb(2)-.5&&mykey(2)<bb(4)+.5)
              inannotatedbb=true;
            end
          %end
        end
        %if(i==5),keyboard;end
        if(dist<.05&&~used{usedidx}(mykey(3)))
          %keyboard
          if(~used{usedidx}(mykey(3)))
            correcterrscore{i}(pk,1)=1;
          end
          used{usedidx}(mykey(3))=1;
        elseif(inannotatedbb)
          correcterrscore{i}(pk,2)=1;
        else
          %mykey=repmat(mykey,size(bbs{i}.x1));
          %if(~any(mykey(1)>bbs{i}.x1-.5&mykey(1)<bbs{i}.x2+.5&mykey(2)>bbs{i}.y1-.5&mykey(2)<bbs{i}.y2+.5))
            correcterrscore{i}(pk,2)=1;
          %end

        end
      end
    end
    
  end
  allvals=cell2mat(correcterrscore);
  [~,ord]=sort(allvals(:,3),'descend');
  prec=cumsum(allvals(ord,1))./cumsum(allvals(ord,1)+allvals(ord,2));
  rec=cumsum(allvals(ord,1))./ngtkeys;
  catch ex,dsprinterr;end
end
