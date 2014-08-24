function idx=sfigure(idx)
  if ishandle(idx)
    set(0,'CurrentFigure',idx);
  else
    idx=figure(idx);
  end
end
