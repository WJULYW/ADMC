For MPNet
go to repalce the content of present /configure/config.py by /configure/configMPNet.py
run ADMC.py with Translation as data augmentation method

For MiniLM
go to repalce the content of present /configure/config.py by /configure/configMniLM.py
run ADMC.py with Translation as data augmentation method

If you want to directly load checkpoint we provide without training, please choose test after you run the ADMC.py and input the name of checkpoint file
checkpoint link: https://drive.google.com/drive/folders/1mzGWqEeiicXlMAoTdh46gkypSGZG5dXH?usp=sharing

Note: 
1. there is a bug in the package ‘google_trans_new’, please fix the bug by following the instructions in   'https://github.com/lushan88a/google_trans_new/issues/36'.
2. When loading the checkpoint file you want, please notice that the pretrain model should be same with the 'pretrain_model' in config.py
