# SAT-master-online. 
This is a Pytorch implementation of our "Learning on Attribute-Missing Graphs" in terms of the node attribute completion task.
The codes of link prediction task are on: https://github.com/xuChenSJTU/SAT-master-link-prediction
 
####################################################################################. 
1. It is accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020. 
You can access the paper in: https://ieeexplore.ieee.org/document/9229522 or https://arxiv.org/abs/2011.01623     

2. An old version of our paper is in: https://arxiv.org/abs/1907.09708   

################# Requirements ########################.   
networkx==2.2.   
pytorch>=1.0.   

################# Notes ########################. 
1. This code is an example code for the node attribute completion task of our SAT paper.  
2. Some example backbones and datasets are provided so that you can directly run them.  
3. Note the name "NANG" appears in ours codes is exactly the name of our "SAT" method.  
"NANG" (Node Attribute Neural Generator) is an old name we used in the previous version.  

################# Usage ########################.   
This dir contains:  
1. NANG's code (NANG_main.py). 
With special parameter setting in NANG_main.py we can get the NANG-no cross, NANG-no self and NANG-no adver in our paper. 

2. Examples of baseline codes,  
(baseline VAE:VAE_main.py, baseline GCN: GCN_main.py, baseline GAT: GCN_main.py, baseline NeighAggre: NeighAggre_main.py). 

3. The evaluation codes including the profiling and node classification (eva_classification_X.py, eva_classification_AX.py). 

4. Data process (precoess.py in ./data), note this file is only used for steam dataset. For cora, citeseer and pubmed, they 
are processed during training.  

5. Examples of datasets including cora, citeseer, pubmed and steam (in ./data). 

How to use:  
**Step1**: process the data, if you want to do the experiment of Steam, you need to. 
preprocess it first (other datasets are already processed), so check the parameter. 
setting and run the code in ./data/process.py using:  

python process.py. 

If you do not want to run the codes on steam, you can directly skip step 1 to step 2.  

**Step2**: Check the parameter setting in NANG_main.py,   
and the parameters used in our paper is as following:  
-------------------------------Cora---------------------------    
topK_list = [10, 20, 50].   
args.dataset='cora'.   
args.epoches=1000.   
args.lr=0.005.   
args.enc_name='GCN' or 'GAT' # choose the encoder for NANG.   
args.alpha=0.2.   
args.dropout=0.5.   
args.patience=100.   
args.neg_times=1.   
args.n_gene=2  # G step.   
args.n_disc=1  # D step.   
args.lambda_recon=1.0  # for self stream    
args.lambda_cross=10.0 # for cross stream.   
args.lambda_gan=1.0   # this parameter is always 1.0, you can ignore it.   

-------------------------------Citeseer---------------------------    
topK_list = [10, 20, 50].   
args.dataset='citeseer'.   
args.epoches=1000.   
args.lr=0.005.   
args.enc_name='GCN' or 'GAT' # choose the encoder for NANG.   
args.alpha=0.2.   
args.dropout=0.5.   
args.patience=100.   
args.neg_times=1.   
args.n_gene=2  # G step.   
args.n_disc=1  # D step.   
args.lambda_recon=1.0  # for self stream      
args.lambda_cross=10.0 # for cross stream.   
args.lambda_gan=1.0   # this parameter is always 1.0, you can ignore it.   

-------------------------------Pubmed---------------------------    
topK_list = [10, 20, 50] # this parameter will be useless for Pubmed.   
args.dataset='pubmed'.   
args.epoches=1000.   
args.lr=0.005.   
args.enc_name='GCN' or 'GAT' # choose the encoder for NANG.   
args.alpha=0.2.   
args.dropout=0.5.   
args.patience=100.   
args.neg_times=1.   
args.n_gene=5  # G step.   
args.n_disc=1  # D step.   
args.lambda_recon=1.0  # for self stream    
args.lambda_cross=50.0 # for cross stream.   
args.lambda_gan=1.0   # this parameter is always 1.0, you can ignore it.   

-------------------------------Steam---------------------------  
topK_list = [3, 5, 10].   
args.dataset='steam'.   
args.epoches=1000.   
args.lr=0.005.   
args.enc_name='GCN' or 'GAT' # choose the encoder for NANG.   
args.alpha=0.2    
args.dropout=0.5.   
args.patience=100.   
args.neg_times=1.   
args.n_gene=5  # G step.   
args.n_disc=1  # D step.   
args.lambda_recon=1.0  # for self stream.   
args.lambda_cross=10.0 # for cross stream.   
args.lambda_gan=1.0   # this parameter is always 1.0, you can ignore it.   

If you get the right parameter setting, you can run the code using:    

GPU mode:    
CUDA_VISIBLE_DEVICES=GPU_num python NANG_main.py    

CPU mode:    
CUDA_VISIBLE_DEVICES=<space> python NANG_main.py   

**Step3**: After you run NANG_main.py, for Cora, Citeseer and Steam, it will directly print the profiling results.    
For Cora, Citeseer and Pubmed, it will save the generated attributes.    
So if you want to make evaluation for the generated attributes with node classification task, check the parameter settings in 
eva_classification_X.py and eva_classification_AX.py, then run it by using:    

Node Classification with Only Attributes (GPU mode):    
CUDA_VISIBLE_DEVICES=GPU_num python eva_classification_X.py   

Node Classification with Both Attributes and Structures (GPU mode):    
CUDA_VISIBLE_DEVICES=GPU_num python eva_classification_AX.py   

#####################Other things#############################. 
1. If you want to run the baseline codes, follow similar steps.  

2. Note that OpenNE contains the codes for DeepWalk and LINE,   
if you want to run them you need to creat necessary soft links which point to files in ./data,   
./features, ./output, ./evaluation.py, ./utils.py, ./figures. 

3. Thanks for your interest in our paper. I am busy in these days, I would provide more related codes when I am free.    

4. If you find the codes or dataset (i.e. steam) useful, please cite our paper, Thank you!  
@article{chen2020learning,  
  title={Learning on Attribute-Missing Graphs},  
  author={Xu Chen and Siheng Chen and Jiangchao Yao and Huangjie Zheng and Ya Zhang and Ivor W Tsang},  
  journal={IEEE transactions on pattern analysis and machine intelligence},  
  year={2020},  
  publisher={IEEE}  
}  

