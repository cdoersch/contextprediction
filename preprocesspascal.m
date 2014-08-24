% Author: Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% Generate an imgs structure and save it to gbz.datasetname.
% It reads all directories contained in gbz.imagedir and
% collects all of the images.
function preprocessindoor67(imagedir,datasetfile)
try
%for(imset=1:2)
  imdirs=struct('name',['JPEGImages']);
  [~,inds]=sort({imdirs.name})
  imdirs=imdirs(inds);
  imgs=[];
  pascalimgs=textread([imagedir '/ImageSets/Main/trainval.txt'],'%s');
  %imdata=[];
  %smdata=[];
  %if(imset==1)
  %  pascalimgs=textread('TrainImages.txt','%s','delimiter','\n');
  %else
  %  pascalimgs=textread('TestImages.txt','%s','delimiter','\n');
  %end
  for(fn=1:numel(imdirs))
    if(strcmp(imdirs(fn).name,'.')||strcmp(imdirs(fn).name,'..'))
      continue;
    end
    imdirs(fn).name
    imgs1=cleandir([imagedir '/' imdirs(fn).name]);
    [~,inds]=sort({imgs1.name});
    imgs1=imgs1(inds);
    rand('seed',fn);
    s=randperm(numel(imgs1));
    imgs2={};
    for(m=1:numel(imgs1))
      if(~ismember(lower([imgs1(m).name(1:end-4)]),lower(pascalimgs)))
        continue;
      end
      [~,pos]=ismember(lower([imgs1(m).name(1:end-4)]),lower(pascalimgs));
      pascalimgs(pos)=[];

      imgs2{end+1}.fullname=[imdirs(fn).name filesep imgs1(m).name];
      imtmp=imread([imagedir '/' imgs2{end}.fullname]);
      sz=size(imtmp);
      imgs2{end}.imsize=sz(1:2);
      if(mod(m,100)==0)
        disp([num2str(m) '/' num2str(numel(imgs1))]);
      end
    end
    imgs=[imgs;cell2mat(imgs2')];
    %imdata=[imdata; imdata1];
    %smdata=[smdata;smdata1];
  end
  imgs=str2effstr(imgs);
  %labelnames=sort(unique(imgs.label));
  %[~,imgs.label]=ismember(imgs.label,labelnames);
  save(datasetfile,'imgs');
%end
catch ex,dsprinterr;end
