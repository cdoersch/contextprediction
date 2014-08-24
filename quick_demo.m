myaddpath;

if(~exist(['quick_demo.mat'],'file'))
  val=input('warning: need to fetch a pre-trained stuff model (approx 600MB) to the current directory.  continue (y/n)?','s');
  if(strcmp(val,'y'))
    disp('fetching...');
    urlwrite('http://graphics.cs.cmu.edu/projects/contextPrediction/quick_demo.mat','./quick_demo.mat');
  else
    disp('answer was not ''y''--aborting.');
    return
  end
end

load('quick_demo.mat');
conf.queryimage=queryimage;
conf.predictorimages=predictorimages;
conf.disp=true;

% This is a prediction that was done near the end of clustering these cars; hence,
% we have a good set of cars to use as predictor images, and all of those cars 
% come with estimates of the region containing the car.
[hm,bghm,~,~,~,~,~,predicted]=contextpredict(querybbox,predictorbboxes,stuffmodelgmm,certaintymap,conf);
% Alternatively, you can run the prediction without any prior knowledge 
% about where the cars are in the predictor images.  Note that the score drops.
%[hm,bghm,~,~,~,~,~,predicted]=contextpredict(querybbox,predictorbboxes,stuffmodelgmm,repmat({[]},size(predictorbboxes,1),1),conf);

probs=min(10,max(-10,hm-bghm));
prob=sum(probs(predicted>.3));
disp(['final log likelihood ratio:' num2str(prob)]);
