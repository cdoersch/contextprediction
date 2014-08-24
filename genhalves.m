% Written by Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% Generate a series of half-diamonds of 1's in a matrix of zeros
% of the specified width.  Used to generate the shape of the dictionaries.
function res=genhalves(width,stepwidth)
  if(~exist('stepwidth','var'))
    stepwidth=width;
  end
  [ptsx ptsy]=meshgrid(1:width,1:width);
  pts=[ptsy(:) ptsx(:)];
  pts(ceil(width*width/2),:)=[];
  pts(pts(:,1)+pts(:,2)<=ceil(width/2),:)=[];
  pts(pts(:,1)+pts(:,2)>ceil(3*width/2),:)=[];
  pts(pts(:,1)-pts(:,2)>=ceil(width/2),:)=[];
  pts(pts(:,2)-pts(:,1)>=ceil(width/2),:)=[];
  pts=[pts ones(size(pts(:,1)))];
  for(i=1:floor(stepwidth/2))
    pix=[ceil(stepwidth/2)-i+1 i]+(width-stepwidth)/2
    hplane=cross([pix 1],[ceil(width/2) ceil(width/2),1]);
    pts2=pts(pts*hplane'>=0,:);
    patch=false(width,width);
    patch(sub2ind([width width],pts2(:,1),pts2(:,2)))=true;
    for(rot=1:4)
      res(:,:,4*(i-1)+rot)=rot90(patch,rot);
    end
  end
end
