function [im] = HOGpicture(w, bs)

if ~exist('bs','var')
  bs = 20;
end
% HOGpicture(w, bs)
% Make picture of positive HOG weights.

% construct a "glyph" for each orientaion
bim1 = zeros(bs, bs);
bim1(:,round(bs/2):round(bs/2)+1) = 1;
bim = zeros([size(bim1) 9]);
bim(:,:,1) = bim1;
for i = 2:9,
  bim(:,:,i) = imrotate(bim1, -(i-1)*20, 'crop');
end

% make pictures of positive weights bs adding up weighted glyphs
s = size(w);
w(w < 0) = 0;
im = zeros(bs*s(1), bs*s(2),floor(size(w,3)/27));
for i = 1:s(1),
  iis = (i-1)*bs+1:i*bs;
  for j = 1:s(2),
    jjs = (j-1)*bs+1:j*bs;
    for k = 1:9,
      for l=1:size(im,3),
      im(iis,jjs,l) = im(iis,jjs,l) + bim(:,:,k) * w(i,j,k+18+27*(l-1));
      %if(nargout==2)
      %  im2(iis,jjs) = im2(iis,jjs) + bim(:,:,k) * w(i,j,k+27+18);
      end
    end
  end
end
