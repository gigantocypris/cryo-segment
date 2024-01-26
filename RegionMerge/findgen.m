function [genx]=findgen(x)
global gen
if gen(x)~=x
      genx=findgen(gen(x));
      gen(x)=genx;
  else
      genx=x;
  end
end