% Written by Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% Initialize the corresp structure.  idx is the upper-left
% corner of the detection in the query image; sz is its height
% and width.  pyrs contains the features for the prediction
% images, and pyridx gives the location in each of these
% pyramids where idx corresponds to.  
%
% The resulting corresp simply defines a correspondence
% between each cell in the detection for the query image and
% the corresponding cells in the predictor images, with a covariance
% of [2 0;0 2] for each Gaussian.
function [corresp]=initcorresp(hogim,idx,sz,pyrs,pyridx);
try
  corresp=cell(size(hogim(:,:,1)));
  for(i=1:numel(pyrs))
    for(y=1:sz(1))
      for(x=1:sz(2))
        if(idx(1)+y-1<size(corresp,1)&&idx(2)+x-1<size(corresp,2))
          corresp{idx(1)+y-1,idx(2)+x-1}=[corresp{idx(1)+y-1,idx(2)+x-1};struct('mu',[pyridx(i,2)+x-1,pyridx(i,1)+y-1],'covar',[2 0 0 2],'level',pyridx(i,3))];
        end
      end
    end
  end
  for(i=1:numel(corresp))
    if(~isempty(corresp{i}))
      corresp{i}=str2effstr(corresp{i});
      corresp{i}.mu=corresp{i}.mu';
      corresp{i}.covar=corresp{i}.covar';
    end
  end
catch ex, dsprinterr;end
end
