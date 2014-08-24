% Written by Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% Given an image (size(im)=[rows cols n]), pad it with the given color
% (n-by-1) in every direction by pad pixels.
function res=padarraycolor(im,pad,color)
  res=repmat(permute(c(color),[2,3,1]),size(im,1)+2*pad,size(im,2)+2*pad);
  res(pad+1:end-pad,pad+1:end-pad,:)=im;
end
