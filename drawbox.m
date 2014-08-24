% Written by Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% Simple utility to draw a box in an image.
% pos is [x1 y1 x2 y2]; width is line thickness
% in pixels
function im=drawbox(im,pos,color,width)
  pos=round(pos);
  inpos=[pos(1:2)+width,pos(3:4)-width];
  inpos(1:2)=max(1,inpos(1:2));
  pos(1:2)=max(1,pos(1:2));
  inpos([4 3])=min(size(im(:,:,1)),inpos([4 3]));
  pos([4 3])=min(size(im(:,:,1)),pos([4 3]));
  tmpim=im(inpos(2):inpos(4),inpos(1):inpos(3),:);
  im(pos(2):pos(4),pos(1):pos(3),:)=repmat(permute(c(color),[2 3 1]),[pos(4)-pos(2)+1,pos(3)-pos(1)+1,1]);
  im(inpos(2):inpos(4),inpos(1):inpos(3),:)=tmpim;
end
