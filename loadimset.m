% Author: Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% Load the image set specified by idx into dswork.  If needed,
% it calls globalz(idx) and stores the results to ds.conf.gbz{idx},
% and loads the imgs structure from the file specified in datasetname
% into ds.imgs{idx}.  
function loadimset(idx)
global ds
origdir=dspwd;
dsup('ds.conf.currimset',idx)
dscd('.ds');
%dsup('ds.conf.currimset',idx)
if(~dsfield(ds,'conf','gbz')||numel(ds.conf.gbz)<idx||isempty(ds.conf.gbz{idx}))
  ds.conf.gbz{ds.conf.currimset}=globalz(ds.conf.currimset);
end
if((~isfield(ds,'imgs'))||numel(ds.imgs)<ds.conf.currimset||isempty(ds.imgs{ds.conf.currimset}))
  load(ds.conf.gbz{ds.conf.currimset}.datasetname);
  if(numel(imgs)>1)
    imgs=str2effstr(imgs);
  end
  if(exist('labelnames','var'))
    ds.conf.gbz{ds.conf.currimset}.labelnames=labelnames;
  elseif(isfield(imgs,'label'))
    ds.conf.gbz{ds.conf.currimset}.labelnames=sort(unique(imgs.label));
    imgs.label=idxof(imgs.label,ds.conf.gbz{ds.conf.currimset}.labelnames);
  end
  if(~isfield(imgs(1),'imsize'))
    for(i=1:numel(imgs))
      imgs(i).imsize=size(imread([ds.conf.gbz{ds.conf.currimset}.cutoutdir imgs(i).fullname]));
      imgs(i).imsize=imgs(i).imsize(1:2);
      if(mod(i,100)==0)
        disp(['loading image sizes: ' num2str(i)]);
      end
    end
    save(ds.conf.gbz{ds.conf.currimset}.datasetname,'imgs');
  end
  ds.imgs{ds.conf.currimset}=imgs;
  if(exist('bboxes','var'))
    ds.bboxes{ds.conf.currimset}=bboxes;
  end
end
dscd(origdir);
%loadimgs(ds.conf.gbz.datasetname,ds.conf.gbz{ds.conf.currimset});

