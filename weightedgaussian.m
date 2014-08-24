% Written by Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% Given a set of weighted datapoints, fit a Gaussian.
function [mu,covar]=weightedgaussian(obs,wts);
      obsweighted=bsxfun(@times,obs,wts);
      mu=sum(obsweighted,1)./sum(wts);
      obs = bsxfun(@minus,obs,mu);  % Remove mean
      obsweighted = bsxfun(@times,obs,wts);
      covar = (obs' * obsweighted) / sum(wts);
end
