clear all


%set the dir for the data.
filename = 'mon_t2_trimmed_rec_nadunet_membrane.mrc';
filename = 'mon_t2_trimmed_rec_nadunet_membrane.mrc_2D_150_clean.mrc';

%filename = 'EPL_dimers_t3_rec_nadunet_membrane.mrc';
filedir=['../2.U-net-new/Pytorch-UNet-master/',filename];
filedir=['./',filename];


%need to add the path for ''EMIODist2'' which is used for reading and writing
% *.mrc files.
addpath(genpath('EMIODist2'));
data=ReadMRC(filedir);

global gen
global m
global n
global nz

    gen=zeros(numel(data),1)+1;
    gen=cumsum(gen);

    indexgen=zeros(numel(data),1);
    [n,m,nz]=size(data);
    dir=zeros(26,3);
    k=0;
    for x=-1:1
        for y=-1:1
            for z=-1:1
                if (x~=0) || (y~=0) || (z~=0)
                    k=k+1;
                    dir(k,:)=[x,y,z;];
                end
            end
        end
    end
    
    %dir=[1,0,0;0,1,0;0,0,1;-1,0,0;0,-1,0;0,0,-1;1,1,0;1,-1,0;-1,1;-1,-1];
  for z=1:nz 
    disp(z)  
    for x=1:n
        for y=1:m
            %disp([z,x,y])
            if data(x,y,z)>0
                ind1=x+(y-1)*n+(z-1)*n*m;
                for q=1:size(dir,1)
                    newz=z+dir(q,1);
                    newx=x+dir(q,2);
                    newy=y+dir(q,3);
                    if newx>0 && newx<=n && newy>0 && newy<=m && newz>0 && newz<=nz
                        if data(x,y,z)*data(newx,newy,newz)~=0
                            ind2=newx+(newy-1)*n+(newz-1)*n*m;
                            unionset(ind1,ind2);
                        end
                    end
                end
                indexgen(ind1)=1;
            end
        end
    end
  end
    index_list=find(indexgen==1);
    for i=1:numel(index_list)
            disp([2,index_list(i)])
            see=findgen(index_list(i));
    end
    
    
    good=zeros(100,1);
    k=0;
    for i=1:numel(index_list)
        disp([3,index_list(i)])
        if gen(index_list(i))~=index_list(i)
            k=k+1;
            good(k)=gen(index_list(i));
        end
    end
% 
    good=unique(good);
    k=numel(good);
    good_cal=zeros(k,1);
    
    index_gen=gen(index_list);

    for i=1:k
        good_cal(i)=numel(find(index_gen==good(i)));
    end
    [see,ind]=sort(good_cal,'descend');
    good_sort=good(ind);
    cal_sort=good_cal(ind);
    
    data_ans=zeros(size(data));
    figure()
    for i=1:10
    see=index_list(find(index_gen==good_sort(i)));
    see3=transzxy3D(see);
    scatter3(see3(:,1),see3(:,2),see3(:,3))
        for q=1:length(see3)
            data_ans(see3(q,1),see3(q,2),see3(q,3))=100;
        end
    hold on
    end
    
    WriteMRC(data_ans,1,[filename,'_3D.mrc'],2,size(data,3))
    
    data_ans=zeros(size(data));
    figure()
    for i=1:10
    see=index_list(find(index_gen==good_sort(i)));
    see3=transzxy3D(see);
    scatter3(see3(:,1),see3(:,2),see3(:,3))
        for q=1:length(see3)
            data_ans(see3(q,1),see3(q,2),see3(q,3))=i;
        end
    hold on
    end
    
    
 WriteMRC(data_ans,1,[filename,'_3D_color.mrc'],2,size(data,3))
    

    


    
    
