% Written by Carl DOersch (cdoersch at cs dot cmu dot edu)
%
% Generate a diamond pattern of 1's in a sqaure
% matrix of zero of the specified width.
function res=genhalves(width)
  [ptsx ptsy]=meshgrid(1:width,1:width);
  pts=[ptsy(:) ptsx(:)];
  pts(ceil(width*width/2),:)=[];
  pts(pts(:,1)+pts(:,2)<=ceil(width/2),:)=[];
  pts(pts(:,1)+pts(:,2)>ceil(3*width/2),:)=[];
  pts(pts(:,1)-pts(:,2)>=ceil(width/2),:)=[];
  pts(pts(:,2)-pts(:,1)>=ceil(width/2),:)=[];
  res=false(width,width);
  res(sub2ind([width width],pts(:,1),pts(:,2)))=true;
end
