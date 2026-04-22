# TriAlignNet
## TriAlignNet: A Triple-Path Cross-Modality Alignment Framework for Multimodal Time Series Forecasting


## Overall Architecture  

The overall architecture of TriAlignNet. The model employs a three-level alignment strategy to integrate information from numerical and textual modalities. Specifically: (1) numerical and textual features are encoded using an MLP and a pretrained LLM, respectively; (2) multimodal alignment is achieved at the distribution level via the MMD loss; (3) a shared learnable anchor matrix is introduced to align multimodal features at the semantic level; and (4) finally, multimodal feature fusion is accomplished through time-text similarity retrieval.  

<img width="4631" height="3481" alt="TriAlignNet" src="https://github.com/user-attachments/assets/98807c45-7cfa-4927-ab2b-4130e4d616c1" />


## Main Result of Multimodal Forecasting

Table 1: Multimodal series forecasting results on eleven datasets from different fields. A lower MSE and MAE indicates better performance, and the best results are highlighted in bold. All results are from four different forecasting horizons: 𝑇={6, 12, 18, 24} for the input sequence length 24.  
<img width="2474" height="1747" alt="单模态主实验结果" src="https://github.com/user-attachments/assets/607ad41c-8110-4a72-8a32-389a1b47af84" />


Table 2: All models were evaluated in a uniform setting, where the input sequence length was fixed to 24, and the results were averaged over the predicted lengths 6, 12, 18, and 24. The optimal and suboptimal results are highlighted in bold and underlined, respectively. "Boost (%)" represents the relative boost of TriAlignNet over the optimal baseline model.  

<img width="2481" height="2931" alt="多模态基线详细实验结果" src="https://github.com/user-attachments/assets/d04bca29-1b97-483a-8a1e-72f8300073ac" />


## Getting Started  
1. Install requirements.  
2. Download data. You can download all the datasets from Google Drive Google Drive [https://drive.google.com/drive/folders/1KCG503FllsoSFHn7IaolrrYZ5NQRggpx?usp=sharing]. Create a seperate folder ./data and put all the csv files in the directory.  
3. Training. All the scripts are in the directory ./scripts/. For example, if you want to get the multivariate forecasting results for ETTh1 dataset, just run the following command, and you can open ./result.txt to see the results once the training is done:  
sh ./scripts/main_forecast.sh

You can adjust the hyperparameters based on your needs (e.g. different length, different look-back windows and prediction lengths.).  

### Training Process  
#### Optimization  
The model is optimized using the Adam optimizer, with a learning rate scheduler to adjust the learning rate dynamically during training. Specifically, during training, we only need to provide an initial learning rate. At the end of each epoch, the learning rate is reduced by half, and training continues.   

## Acknowledgement  

We appreciate the following github repo very much for the valuable code base and datasets:  
TimeCMA: https://github.com/ChenxiLiu-HNU/TimeCMA  
DMMV: https://github.com/D2I-Group/dmmv  
FreqLLM: https://github.com/biya0105/FreqLLM  

## Contact  
If you have any questions or concerns, please contact us: yejunjie@stu.yun.edu.cn or zhaochunna@ynu.edu.cn or submit an issue.

