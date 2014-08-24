% given a matrix and a subscript, determine
% if the subscript is within the bounds of the matrix
function res=valididx(mat,varargin)
  inds=cell2mat(varargin);
  sz=size(mat);
  res=all(inds>0 & inds<=sz);
end
