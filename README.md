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

- The Myo Armband dataset used for offline pre-training can be accessed from [GitHub](https://github.com/UlysseCoteAllard/MyoArmbandDataset).

- The dataset utilized for meta-learning can be downloaded from [IEEE DataPort](https://ieee-dataport.org/documents/emg-eeg-dataset-upper-limb-gesture-classification).

- The FlexAdapt dataset, which is used for knowledge distillation and on-device fine-tuning, can be accessed from [HuggingFace](https://huggingface.co/datasets/deremustapha/FlexAdapt_EMG_Dataset).
