% Written by Carl Doersch (cdoersch at cs dot cmu dot edu)
%
% Given a set of patch bounding boxes, verify whether they depict the same object
% using contextual information.  
%
% mydets: the set of patch bounding boxes, in standard matrix form.  
%
% stuffmodel: the stuff model.  See the comment in contextpredict.m
%
% state: If previous predictions have been made with this particular set
%        of detections, you can pass in the state returned from that previous
%        run.  This captures the entire internal state of verifyelementdist,
%        so if you exited after only a few predictions, passing in the state
%        will cause verifyelementdist to pick up where it left off.  If no such
%        state exists, pass [].
%
% npreds: the maximum number of predictions (i.e. calls to contextpredict.m)
%         to make before returning.
%
% conf: additional configuration options to pass to contextpredict.
function [score,state]=verifyelementdist(mydets,stuffmodel,state,npreds,conf)
  try
  global ds;
  if(~exist('conf','var'))
    conf=struct();
  end
  if(isfield(state,'roundid'))
    roundid=state.roundid;
  else
    roundid=1;
  end
  roundsremaining=npreds;
  % select only the best detection for each image; duplicates within
  % the same image can cause problems since scenes tend to be symmetric.
  topdets=cell2mat(maxkall(distributeby(mydets,mydets(:,7)),5,1));
  isvalid=ismember(mydets,topdets,'rows');
  validinds=find(isvalid);
  if(roundid==1)
    state.certaintymap=cell(size(mydets,1),1);
    % For the first prediction, we trust the input ordering; we predict the first one in the order based on the next 20.
    [hm,bghm,corresp,pyrs,inimpyrlevel,blurconf,usemap,predicted,state.preds{1}.have_preds_orig,state.preds{1}.predictednooc]=contextpredict(mydets(1,:),mydets(validinds(2:21),:),stuffmodel,state.certaintymap,conf);
    % Now store everything we computed in the internal state.
    state.certaintymap{1}=exp(hm)./(exp(hm)+exp(bghm));;
    state.probs=zeros(size(mydets(:,1)))*NaN;
    state.thresh=state.probs;
    state.selectionscores=zeros(size(mydets(:,1)));
    state.selectionn=zeros(size(mydets(:,1)));
    state.selectionn(validinds(2:21))=1;
    state.chosen=false(size(mydets(:,1)));
    [state.probs(1),state.selectionscores(validinds(2:21))]=computeCertainty(hm,bghm,predicted,.3,usemap,blurconf);
    state.thresh(1)=.3;
    state.preds{1}.hm=hm;
    state.preds{1}.bghm=bghm;
    state.preds{1}.inimpyrlevel=inimpyrlevel;
    state.preds{1}.usemap=usemap;
    state.preds{1}.predicted=predicted;
    state.allpredinds{1}=validinds(2:21);
    roundid=roundid+1;
    roundsremaining=roundsremaining-1;
    state.chosen(1)=true;
  end

  % To do the later predictions, we need to compute LDA scores.  Hence we need to extract
  % patches.
  featlib=extractpatches(mydets,[],struct('explicitsize',ds.conf.params.patchCanonicalSize,'featurize',true));
  while(true)
    if(exist('verifystop','file'))
      keyboard
    end

    % To choose the next query and predictor patches, we take the
    % verified patches and use them as 'positive' for the LDA.
    % This means we get a slightly different ordering each time, which
    % is actually pretty important, since it means that more patches
    % get a usage score computed, which can ultimately help us find
    % new instances far down in the original E-LDA ranking.
    if(sum(state.probs>0)>1)
      template=mean(featlib(state.probs>0,:),1);
      ldascore=featlib*c(template);
    else
      ldascore=mydets(:,5);
    end
    ldascore(~isvalid)=-Inf;
    ldascore(state.chosen)=-Inf;
    % This is the 'usage score'--i.e. how much the patch
    % contributed to the correct predictions.
    predscore=state.selectionscores;
    predscore(state.chosen)=-Inf;
    predscore(~isvalid)=-Inf;
    prevpreds=state.probs;
    prevpreds(isnan(prevpreds))=-Inf;
    [~,runidx]=max(predscore);
    ldascore(runidx)=-Inf;
    [scr,prevpredidx]=maxk(prevpreds,10);
    if(roundsremaining==0)
      break;
    end
    prevpredidx(scr<0)=[];
    [scr,ldaidx]=sort(ldascore,'descend');
    ldaidx(scr==-Inf)=[];
    ldaidx(ismember(ldaidx,prevpredidx))=[];
    ldaidx=ldaidx(1:min(numel(ldaidx),20-numel(prevpredidx)));
    % Get upt to half from the ones with high usage score, and the
    % rest from the ones with high lda score.
    predidx=[c(prevpredidx);c(ldaidx)];

    state.allpredinds{runidx}=predidx;
    if(numel(state.certaintymap)<max(state.allpredinds{runidx}))
      state.certaintymap{max(state.allpredinds{runidx}),1}=[];
    end
    [hm,bghm,corresp,pyrs,inimpyrlevel,blurconf,usemap,predicted,state.preds{runidx}.have_preds_orig,state.preds{runidx}.predictednooc]=contextpredict(mydets(runidx,:),mydets(state.allpredinds{runidx},:),stuffmodel,state.certaintymap(state.allpredinds{runidx}),conf);
    state.certaintymap{runidx}=exp(hm)./(exp(hm)+exp(bghm));
    state.selectionn(state.allpredinds{runidx})=state.selectionn(state.allpredinds{runidx})+1;
    [state.probs(runidx),selscr]=computeCertainty(hm,bghm,predicted,.3,usemap,blurconf);
    state.selectionscores(state.allpredinds{runidx})=state.selectionscores(state.allpredinds{runidx})+selscr;
    state.thresh(runidx)=.3;

    state.preds{runidx}.hm=hm;
    state.preds{runidx}.bghm=bghm;
    state.preds{runidx}.inimpyrlevel=inimpyrlevel;
    state.preds{runidx}.usemap=usemap;
    state.preds{runidx}.predicted=predicted;
    roundid=roundid+1;
    roundsremaining=roundsremaining-1;
    state.chosen(runidx)=true;
  end
  state.roundid=roundid;
  score=c(state.probs);
  catch ex,dsprinterr;end
end

function [prob,usagescore]=computeCertainty(hm,bghm,predicted,thresh,usemap,blurconf)
  probs=min(10,max(-10,hm-bghm));
  prob=sum(probs(predicted>thresh));
  if(nargout>1)
    usagescore=c(nansum(nansum(bsxfun(@times,usemap,blurconf.*(predicted>thresh)),1),2));
  end
end
