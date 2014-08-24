% Written by Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% Generates a display of predicted keypoints.  Requires access
% to several variables in the workspace at the end of objectdiscovery_main.m,
% but does not modify any of them.
for(dsidx=numel(ds.kps):-1:1)
  disp(dsidx)
  if(isempty(ds.kps{dsidx}))
    continue
  end
  ok=find(ds.kps{dsidx}(:,4)>200);
  if(numel(ok)<5),continue;end
  fig=figure(1);
  clf
  subplot(2,1,1);
  imagesc(getimg(ds.runimgs(dsidx)));
  hold on;
  axis equal;
  plot(ds.kps{dsidx}(ok,1),ds.kps{dsidx}(ok,2),'rx');
  labs={};
  for(i=1:numel(ok))%size(ds.kps{dsidx},1))
    labs{i}=[part_names{ds.kps{dsidx}(ok(i),3)} num2str(ds.kps{dsidx}(ok(i),4))];
    disp(labs{i});
  end
  text(ds.kps{dsidx}(ok,1),ds.kps{dsidx}(ok,2),labs,'VerticalAlignment','bottom','HorizontalAlignment','left','Color','r');
  set(gca,'YTick',[]);
  set(gca,'XTick',[]);
  box off
  % look up the keypoints from other images that were used
  % to predict each of the keypoints in this image.  Extract a small
  % patch aroudn them and display it.
  grabbox=[];
  for(i=1:numel(ok))
    mykp=ds.kps{dsidx}(ok(i),:);
    annotimidx=ds.annotimidx(mykp(7));
    imkps=ds.annot(mykp(7));
    pt=imkps.keypoints(mykp(6),1:2);
    bb=imkps.bbox;
    sz=max(bb(3)-bb(1)+1,bb(4)-bb(2)+1)/6;
    grabbox(i,1:8)=[pt-sz,pt+sz,0,0,annotimidx,mykp(8)];
  end
  pats=extractpatches(grabbox,[],struct('noresize',true));
  for(i=1:numel(pats))
    pats{i}=im2double(pats{i});
    subplot(20,14,14*10+(i));
    imagesc(pats{i});
    set(gca,'YTick',[]);
    set(gca,'XTick',[]);
  axis equal;
  box off
    subplot(20,14,14*11+(i));
    %subplot(14,20,12+(i-1)*20);
    frac=ds.kps{dsidx}(ok(i),4)/max(ds.kps{dsidx}(:,4));
    imagesc(pats{i}*frac+(pats{i}*0+1)*(1-frac));
    set(gca,'YTick',[]);
    set(gca,'XTick',[]);
  axis equal;
  box off
  end
  waitforbuttonpress
end
