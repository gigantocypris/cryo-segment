# cryo-segment

## Quick start: https://chanzuckerberg.github.io/cryoet-data-portal/cryoet_data_portal_docsite_quick_start.html

Log on to Perlmutter
> cd /global/cfs/cdirs/m3562/users/vidyagan
> git clone https://github.com/vganapati/cryo-segment.git

> module load python
> module load pytorch/2.1.0-cu12
> conda create -n cryo
> conda activate cryo

> pip install -U cryoet-data-portal
> pip install -U mrcfile


Opening MRC files
https://mrcfile.readthedocs.io/en/stable/usage_guide.html

Data Portal GitHub:
https://github.com/chanzuckerberg/cryoet-data-portal/tree/main

## Getting Started after Setup:

Log on to Perlmutter
"Take" data:
> cd /global/cfs/cdirs/m3562/users/vidyagan/output-cryo-segment
> take -u nksauter emd_10439.map.gz

> module load python
> module load pytorch/2.1.0-cu12
> conda activate cryo
> cd /global/cfs/cdirs/m3562/users/vidyagan/cryo-segment

Running the Zhou et al pipeline:

> mkdir checkpoint
> mkdir ciro
> cp Zhou_2023/UNET/data/files/*.txt ciro/
> python Zhou_2023/UNET/data/createcirosimnewpart.py (make sure line 118 is "dir_name = 'ciro'" creates ciro folder in working directory. copy train.txt, test.txt and labels.txt to this folder, make a "checkpoint" folder in the working directory)
> salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 1 --account=m3562
> python Zhou_2023/UNET/train.py -e NUM_EPOCHS (try 50 epochs to start, 500 epochs for production)

Creating the test images:
Change Line 118 of "Zhou_2023/UNET/data/createcirosimnewpart.py"  to dir_name='val' and run the code to generate the test data. (originally, dir_name = 'ciro')
> python Zhou_2023/UNET/data/createcirosimnewpart.py

Following the README directions:

Run the code "python Zhou_2023/UNET/visunet.py" to test result on one images. Change the image file by change the code:

Line 31:  image_path = 'data/val/200_3_data.png'

(created "result/vis" folder in working directory)
check the results in "result/vis" in the working directory

> python Zhou_2023/UNET/visunet.py

3.3 test a group model on a group images.

Run the code "visgroupunet.py" to test result by group. It will test allChange model in "checkpoint/", Change the image file by change the code:

Line 53: image_list=['100_0_data.png','200_0_data.png','300_0_data.png','400_0_data.png','500_0_data.png','600_0_data.png','700_0_data.png','800_0_data.png','900_0_data.png','1000_0_data.png']

check the results in "result" in working directory

> python Zhou_2023/UNET/visgroupunet.py

3.4 test model on one slice of real membrane files.

Run the code "visunetmen.py",change the configuration as below. 
> python Zhou_2023/UNET/visunetmen.py

Need MRC file:
/global/cfs/cdirs/m3562/users/vidyagan/output-cryo-segment/dataset_10010_TE2
/global/cfs/cdirs/m3562/users/vidyagan/output-cryo-segment/dataset_10010_TE13

Line 31 : model_path_root='checkpoint/' #models directory.
Line 32 : image_path = 'mon_t1_trimmed.rec.nad' #which file used for test.
Line 36 : model_path=model_list[0] #which model used for test.
Line 38 : index=100 #which slice used for test.

# STOPPED HERE
> python Zhou_2023/UNET/visunetmen_mrc.py
3.5 test model on all slices of real membrane files.
Run the code "visunetmen_mrc.py",change the configuration as below. 

Line 31 : model_path_root='checkpoint/' #models directory.
Line 32 : image_path = 'mon_t1_trimmed.rec.nad' #which file used for test.
Line 36 : model_path=model_list[0] #which model used for test.

Next steps:

RL
RegionMerge
GP3D