% Written by Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% The same as imagesc, but return the color image rather than
% displaying it.
function im=myimagesc(im,rng)
  im=double(im);
  if(~exist('rng','var'))
    rng=[min(im(:)) max(im(:))];
  end
  im=im-rng(1);
  im=im/(rng(2)-rng(1));
  im(isnan(im(:)))=0;
  im=heatmap2jet(im);
end
