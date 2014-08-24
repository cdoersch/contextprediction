% Written by Carl Doersch cdoersch@cs.cmu.edu
%
% Given a set n Gaussians in d dimensions (whose parameters are 
% given in the pages of mus and sigmas), and a set of mixing 
% weights that may be interpreted as a Gaussian mixture model,
% return the mean and covariance matrix of the global model.
%
% size(mus)=[1 d n]
% size(sigmas)=[d d n]
% size(alphas)=[n 1] or [1 n]
%
% size(resmu)=[1 d]
% size(ressigma)=[d d]

function [resmu ressigma]=combinegaussians(mus,sigmas,alphas)
  alphas=alphas(:);
  mus=permute(mus,[3,2,1]);
  resmu=alphas'*mus./sum(alphas);
  ressigma=zeros(size(sigmas(:,:,1)));
  for(i=1:numel(alphas))
    ressigma=ressigma+sigmas(:,:,i)*alphas(i)+alphas(i)*(resmu-mus(i,:))*(resmu-mus(i,:))';
  end
end
