% Written by Carl Doersch, cdoersch at cs dot cmu dot edu
%
% Cap probabilities to be less than the value cap, along the dimension dim.  
% res is the same size as vec, but for each vector v along dimension dim, 
% the corresponding vector v2 in res has the following properties:
%
% 1) sum(v2)=1
% 2) there exists some constants k and c such that:
%    i) if v(i) smaller than k, v2(i)=v(i)*c
%    ii) if v(i) greater than or equal to k, then v2(i)=cap
% 
% I'm reasonably confident that given v and cap, there is only one
% v2 satisfying these properties.  But I'm too lazy to prove it.
%
function res=capprobabilities(vec,cap,dim)
try
  permutevec=1:numel(size(vec));
  permutevec(1)=dim;
  permutevec(dim)=1;
  vec=permute(vec,permutevec);
  sz=size(vec);
  vec=reshape(vec,sz(1),[]);
  %n=size(vec,dim);
  [sortvec,sortord]=sort(vec,1);
  idxforsortvec=sub2ind(size(sortord),sortord,repmat(1:size(sortord,2),size(sortord,1),1));
  sumvec=cumsum(sortvec,1);
  ratioforidx=bsxfun(@times,c(size(vec,1)-1:-1:1),sortvec(2:end,:))./sumvec(1:end-1,:);
  limitratio=(c(size(vec,1)-1:-1:1)*cap./(max(0,1-c(size(vec,1)-1:-1:1)*cap)));
  limitratio(isnan(limitratio))=Inf;
  nexttoohigh=bsxfun(@gt,ratioforidx,limitratio);
  nexttoohigh=[false(1,size(nexttoohigh,2));nexttoohigh];
  if(any(c(diff(double(nexttoohigh),1,1))<0))
    error('capprobs invariant failed');
  end
  normfact=sum((~nexttoohigh).*sortvec,1)./(1-cap*sum(nexttoohigh));
  vec=bsxfun(@rdivide,vec,normfact);
  vec(idxforsortvec(nexttoohigh))=cap;
  vec=reshape(vec,sz);
  res=permute(vec,permutevec);
catch ex,dsprinterr;end
end
