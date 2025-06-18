# Adaptive model updates for motor-intent decoding combined with flexible surface electromyography sensor grid

</div>

<div align="center">
  <img src="imgs/1.png"/>
</div> 


## Table of Contents
- [Usage](#usage)
  - [Installation and Setup](#installation)
  - [Datasets](#dataset)
- [Offline pre-training](#offline)
  - [Pre-Training](#step1)
  - [Meta-Learning](#step2)
  - [Knowledge-Distillation](#step3)
- [On-device AdaptiveEdge](#online)



# Usage

<h2 id="installation">🛠 Installation and Setup</h2>

1. Clone the repo:
   ```bash
   git clone https://github.com/deremustapha/AdpativeEdge

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

<h2 id="dataset">📊 Dataset</h2>

- The Myo Armband dataset used for offline pre-training can be accessed from [GitHub](https://github.com/UlysseCoteAllard/MyoArmbandDataset).

- The dataset utilized for meta-learning can be downloaded from [IEEE DataPort](https://ieee-dataport.org/documents/emg-eeg-dataset-upper-limb-gesture-classification).

- The FlexAdapt dataset, which is used for knowledge distillation and on-device fine-tuning, can be accessed from [HuggingFace](https://huggingface.co/datasets/deremustapha/FlexAdapt_EMG_Dataset).



<h2 id="offline"> Offline Pre-training</h2>

During the offline pre-training, it is important that the model weights are saved so they can be used for intermediate offline training; on-device model updates; to enable replication of the result reported in the study. In addition, the correct flags such as the desired model, window-size, overlap, participant, experimental session etc., should be used to ensure the replication of the result.

1. [Pre-Training](#step1):
   ```bash
   python3 pre-train.py

2. [Meta-Learning](#step2):
   ```bash
   python3 meta-learning.py


3. [Knowledge-Distillation](#step3):
   ```bash
   python3 kd.py


<h2 id="online"> On-device AdaptiveEdge Model Update</h2>

Typically, the pre-trained models need to be depolyed on an Ultra96-V2 FPGA for inference. However, a fine-tune script is provided to allow the replication of the AdaptiveEdge and other model update methods on local computer. 

4. [AdaptiveEdge](#step):
   ```bash
   fine-tune.py

