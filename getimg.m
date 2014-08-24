% Author: Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% Load an image at the position specified by idx.  Optionally,
% the first parameter can be the ds structure, avoiding
% a call to global.  
function res=getimg(ds,idx)
  if(nargin<2)
    idx=ds;
    clear ds;
    global ds;
  end
  imgs=dsload('.ds.imgs{ds.conf.currimset}');
  if(numel(imgs.fullname)<idx(1))
    idx(1)=idx(1)-numel(imgs.fullname);
    flip=1;
  else
    flip=0;
  end
  gbz=dsload('.ds.conf.gbz{ds.conf.currimset}');
  res=imread([gbz.cutoutdir imgs.fullname{idx(1)}]); 
  if(flip)
    res=res(:,end:-1:1,:);
  end
  if(dsfield(ds,'conf','params','maxpixels'))
    if(size(res,1)*size(res,2)>ds.conf.params.maxpixels)
      ressz=round([size(res,1),size(res,2)]*sqrt(ds.conf.params.maxpixels/(size(res,1)*size(res,2))))
      res=imresize(res,ressz);
    end
  end
  if(dsfield(ds,'conf','params','minpixels'))
    if(size(res,1)*size(res,2)<ds.conf.params.minpixels)
      ressz=round([size(res,1),size(res,2)]*sqrt(ds.conf.params.minpixels/(size(res,1)*size(res,2))))
      res=imresize(res,ressz);
    end
  end
  if(size(res,3)==1)
    res=cat(3,res,res,res);
  end
  if(numel(idx)==2)
    if(idx(2))
      res=res(:,end:-1:1,:);
    end
  end
end
