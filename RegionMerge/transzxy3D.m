function [cor_xy] = transzxy3D(cod)

global m
global n
q=numel(cod);
cor_xy=zeros(q,3);
temp=cod;
cor_xy(:,3)=floor((temp-1)/(m*n))+1;
temp=mod(temp-1,m*n)+1;
cor_xy(:,2)=floor((temp-1)/(n))+1;
cor_xy(:,1)=mod(temp-1,n)+1;
%scatter(cor_xy(:,1),cor_xy(:,2))
end

