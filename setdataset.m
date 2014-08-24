% set the dataset that the algorithm uses.  In particular, this
% affects the behavior of getimgs and getimg.  
%
% idx is a unique integer to identify the dataset. 
%
% imgs is a struct with the fields:
%   - 'fullname': a cell array of labels, such that 
%     [datadir '/' imgs.fullname{i}] will resolve to the  absolute 
%     path of an image.
%   - 'label': (optional) an index into the labelnames cell array 
%      for the image's label.
%
% imgs will be stored in dswork in the location .ds.imgs{idx}, in memory
% and on disk. idx will be stored in ds.conf.currimset (note that this
% is a relative url; it will only be set for the current directory.  getimg
% reads the value from this location).
%
% datadir: the directory containing all of your data
%
% labelnames: names for each label used in the dataset.
%
% weburl: an optional url that points to the web-accessible
% location of datadir, i.e. such that [imgsurl '/' imgs.fullname{i}]
% lets you download the image.  Relying on this value is not recommended;
% in the context prediction and discriminative mode seeking projects, it
% is used only in html displays.
%
% This information will be stored in .ds.conf.gbz{idx}.  
function setdataset(idx,imgs,datadir,labelnames, weburl)
  global ds;
  dsup('ds.conf.currimset',idx);
  if(nargin>1)
    if(~exist('weburl','var'))
      weburl='';
    end
    if(datadir(end)~=filesep)
      datadir=[datadir filesep];
    end
    gbz=struct('labelnames',{labelnames},'imgsurl',weburl,'cutoutdir',datadir,'ctrelname','');
    mydir=dspwd;
    dscd('.ds');
    dsup(['ds.imgs{' num2str(idx) '}'],imgs);
    dsup(['ds.conf.gbz{' num2str(idx) '}'],gbz);
    dscd(mydir);
    dsup(['ds.conf.gbz{' num2str(idx) '}'],gbz);% TODO: this should really only be kept in 
                                                % the root .ds.conf; relying on the
                                                % non-root ds.conf creates
                                                % unintuitive behavior.
    dssave;
  end
end
