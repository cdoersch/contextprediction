% Written by Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% Scale detections so that the center remains the same, but
% each side is increased by a factor of scalefact.  If maxextent
% is given, it is interpreted as the size of an image, and the
% bounding box is cropped to fit within it.
function dets2=scaledets(dets,scalefact,maxextent);
  part1=(1+scalefact)/2;
  part2=(1-scalefact)/2;
  dets2=dets;
  dets2(:,1)=(dets(:,1)*part1+dets(:,3)*part2);
  dets2(:,2)=(dets(:,2)*part1+dets(:,4)*part2);
  dets2(:,3)=(dets(:,3)*part1+dets(:,1)*part2);
  dets2(:,4)=(dets(:,4)*part1+dets(:,2)*part2);
  if(nargin>2)
    dets2(:,[1 2])=max(dets2(:,[1 2]),1);
    dets2(:,3)=min(dets2(:,3),maxextent(2));
    dets2(:,4)=min(dets2(:,4),maxextent(1));
  end
end
