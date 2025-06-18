# Adaptive model updates for motor-intent decoding combined with flexible surface electromyography sensor grid

</div>

<div align="center">
  <img src="imgs/1.png"/>
</div> 


## Table of Contents
- [Usage](#usage)
  - [Installation and Setup](#installation-and-setup)
  - [Datasets](#datasets)
- [Offline Training](#offline)
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

- The Myodataset used for offline pre-training can be downloaded from [GitHub](https://github.com/UlysseCoteAllard/MyoArmbandDataset).

- The dataset used for meta-learning can be downloaded from [IEEEPort](https://ieee-dataport.org/documents/emg-eeg-dataset-upper-limb-gesture-classification).

- FlexAdapt dataset utilized for knowledge distillation and on-device fine-tuning can be gotten on [HuggingFace](https://huggingface.co/datasets/deremustapha/FlexAdapt_EMG_Dataset)
