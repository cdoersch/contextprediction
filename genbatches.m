% Generate batch indices.  npts is the number of points to be assigned to baches;
% batchsize is the size of each batch.  res is a npts-by-1 vector
% specifying the batch number for each point.
function res=genbatches(npts,batchsz)
  nbatches=ceil(npts/batchsz);
  res=repmat(1:nbatches,batchsz,1);
  res=res(:);
  res=res(1:npts);
end
