% Written by Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% Generates a display of hms and bghms which lets us visualize the clusters.
% The logic is very similar to the computation of blurthingprob, but it has
% some additional logic to make sure that the patch itself is displayed,
% convert the output map into pixel coordinates, crop out the box, and draw
% lines where the image ends.
function imwarp=displaylikelihoodmask(mydets,hms,bghms,predictedsctx,inimpyrlevs,conf)
try
  global ds;
  if(~exist('conf','var'))
    conf=struct();
  end
  for(i=1:numel(hms))
    if(isempty(hms{i}))
      continue;
    end
    pos2=mydets(i,[1:4]);
    if(mydets(i,8))
      pos2([1 3])=-pos2([3 1])+size(getimg(mydets(i,7)),2)+1
    end
    pos2=pos2/2^((inimpyrlevs(i)-1-8)/8)/ds.conf.params.sBins-1;
    pos2(1:2)=floor(pos2(1:2));
    pos2(3:4)=ceil(pos2(3:4));
    pos2=max(1,pos2);
    pos2(3)=min(pos2(3),size(hms{i},2));
    pos2(4)=min(pos2(4),size(hms{i},1));
    badhmpos=hms{i}(pos2(2):pos2(4),pos2(1):pos2(3));
    badpred=predictedsctx{i}(pos2(2):pos2(4),pos2(1):pos2(3));
    badpred(badhmpos==0)=1;
    predictedsctx{i}(pos2(2):pos2(4),pos2(1):pos2(3))=badpred;
    badhmpos(badhmpos==0)=20;
    hms{i}(hms{i}==0)=-20;
    %sfigure(1);
    %imagesc(hms{i});
    hms{i}(pos2(2):pos2(4),pos2(1):pos2(3))=badhmpos;
    %sfigure(2);
    %imagesc(hms{i});
    %keyboard
    certaintymap{i}=exp(hms{i})./(exp(hms{i})+exp(bghms{i}));
    predaccuracy=certaintymap{i};
    onempredaccuracy=exp(bghms{i})./(exp(hms{i})+exp(bghms{i}));
    tmp=exp(gaussfiltervalid(log(predaccuracy),predictedsctx{i}>0,2));
    bgtmp=exp(gaussfiltervalid(log(onempredaccuracy),predictedsctx{i}>0,2));
    pred=predictedsctx{i};
    pred(pred==1)=.999999999;
    onempred=1-pred;
    pred=exp(gaussfiltervalid(log(pred),predictedsctx{i}>0,2));
    onempred=exp(gaussfiltervalid(log(onempred),predictedsctx{i}>0,2));


    bgblur=bgtmp./(tmp+bgtmp);
    fgblur=tmp./(tmp+bgtmp);
    bc=fgblur.*pred;
    if(isfield(conf,'thresh')&&~isnan(conf.thresh(i)))
      bc=bc-3.5*(conf.thresh(i)-.3);
    end
    blurcertainty{i}=min(1,max(0,fgblur.*pred*3.5-.5));%./(fgblur.*pred+bgblur.*onempred);
    if(dsbool(conf,'returnmask'))
      continue;
    end
    pyridx2=i;
    currim=pyridx2;
    im=im2double(getimg(mydets(pyridx2,[7:8])));
    pos3=mydets(i,[1:4]);
    if(mydets(i,8))
      pos3([1 3])=-pos3([3 1])+size(im,2)+1
    end
    pos3=scaledets(pos3,3);
    predsize=round(size(im(:,:,1))/2^((inimpyrlevs(pyridx2)-1-8)/8));
    predsize2=floor(predsize/8)*8;
    tmpsize=round(size(im(:,:,1)).*(predsize2./predsize));
    %certmap=zeros(size(im(:,:,1)));
    tmpmap=imresize(padarray(blurcertainty{currim},[1,1,0],'replicate'),tmpsize,'bicubic');
    %certmap(1:size(tmpmap,1),1:size(tmpmap,2))=tmpmap;
    certmap=tmpmap;
    im=im(1:size(tmpmap,1),1:size(tmpmap,2),:);
    im=bsxfun(@times,im,certmap)+bsxfun(@times,ones(size(im)),(1-certmap));
    boxwidth=round((pos3(3)-pos3(1)+1)/75);
    im=drawbox(im,pos3,[0 0 0],boxwidth);
    pad=max(max(1-min(pos3([1 2])),pos3(4)-size(im,1)),pos3(3)-size(im,2));
    im=padarraycolor(im,boxwidth,[0 0 0]);
    pos3=pos3+boxwidth;

    if(pad>0)
      im=padarraycolor(im,pad,[1 1 1]);
      pos3=pos3+pad;
    end

    %certmap=combineprobmaps(blurcertainty{i},1-blurcertainty{i}
    imwarp{pyridx2}=im(pos3(2):pos3(4),pos3(1):pos3(3),:);%correspwarp(corresp,ds.conf.params.sBins,im,pyrs,pyridx2);
    %certmap2=imresize(padarray(blurconf.*quantile(usemap,.3,3),[1,1,0]),ds.conf.params.sBins*2^((inimpyrlevel-1-8)/8),'bilinear');
    %certmap2=certmap2./max(certmap2(:));
    %certmap2=min(1,certmap2*2);
    %imwarp{pyridx2}=bsxfun(@times,certmap2,imwarp{pyridx2})+bsxfun(@times,(1-certmap2),ones(size(imwarp{pyridx2})));

  end
  if(dsbool(conf,'returnmask'))
    imwarp=blurcertainty;
  end
catch ex,dsprinterr;end

