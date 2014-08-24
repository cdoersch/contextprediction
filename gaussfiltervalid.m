% Written by Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% given an image im and a binary mask valid of the same size, perform
% a gaussian blur on the image with standard deviation sigma.
% Pixels marked as 0 in invalid will be ignored, but this is done in a
% way so that output pixels which should draw input from invalid pixels
% instead draw more heavily from the valid ones.  Mathematically, 
% in a standard gaussian blur, res(i,j)=sum(sum(k.*im)), where k is
% a gaussian bump of the same size as im, where the bump is centered
% at (j,i), and sum(sum(k))==1.  To perform our blurring, we simply
% rescale k to k_new so that sum(sum(k_new.*valid))==1, and compute
% res(i,j)=sum(sum(k_new.*valid.*im)).
function res=gaussfiltervalid(im,valid,sigma)
  im=im.*valid;
  im(~valid)=0;%handle NaN and Inf
  filt=-2*ceil(sigma):2*ceil(sigma);
  filt=exp(-filt.^2/(2*sigma.^2));
  im=conv2(im,filt,'same');
  im=conv2(im,filt','same');
  valid=conv2(double(valid),filt,'same');
  valid=conv2(valid,filt','same');
  res=im./valid;
end
