function []=unionset(x,y)
  global gen
  genx=findgen(x);
  geny=findgen(y);
  if genx~=geny
      gen(genx)=geny;
%      gen(gen==genx)=geny;
  end
end