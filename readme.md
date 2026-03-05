Dataset link to start training directly(2d dataset)-https://drive.google.com/drive/folders/1qUvLPEPlaM3KkNVI1GKTAfc4Z0jz1YuU?usp=drive_link

Original 3d dataset link - https://drive.google.com/file/d/1YZQFSonulXuagMIfbJkZeTFJ6qEUuUxL/view?usp=drive_link
Steps to derieve 2d data from the original 3d dataset:
1 - Extract the Task07_Pancreas.tar file
2 - Set the directory of the extracted folder in the prepare_dataset.py file and run it.
3 - Run split_data.py to get 80/20 or 70/15/15 split , whichever you want.
4 - Run the Unet.ipynb file.

To train the model, open Unet.ipynb file and run each block seperatly.
Keep all the files in the same root directory.
