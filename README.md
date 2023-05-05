Preciser comparison: Augmented multi-layer dynamic contrastive strategy for text2text question classification
====
This is the official code of the paper "Preciser comparison: Augmented multi-layer dynamic contrastive strategy for text2text question classification
"
-------
Train with MPNet
-------
Go to repalce the content of present /configure/config.py by /configure/configMPNet.py<br>
run ADMC.py with Translation as data augmentation method<br>

Train with MiniLM
-------
Go to repalce the content of present /configure/config.py by /configure/configMniLM.py<br>
run ADMC.py with Translation as data augmentation method<br>

Test
-------
If you want to directly load checkpoint we provide without training, please choose test after you run the ADMC.py and input the name of checkpoint file<br>
checkpoint link: https://drive.google.com/drive/folders/1mzGWqEeiicXlMAoTdh46gkypSGZG5dXH?usp=sharing<br>
The option of the pretrain model should follow by above instructions.

Note
-------
1. There is a bug in the package ‘google_trans_new’, please fix the bug by following the instructions in  https://github.com/lushan88a/google_trans_new/issues/36; <br>Or just delete the third line in data_prepare.py and don't use augmentation, while it will make performance not good as checkpoint<br>
2. When loading the checkpoint file you want, please notice that the pretrain model should be same with the 'pretrain_model' in config.py.<br>
3. For now, we provide expanded COVID-Q as an example. Later, we will release other datasets we mentioned in our paper and publicize the GitHub repository.  
