Adapting SAM to SAR Avalanche DetectionOfficial implementation of the paper: "Adapting Segment Anything Model (SAM) for Avalanche Detection in SAR Imagery" by Riccardo Gelato and Carlo Sgaravatti.
📌 Overview
Manual segmentation of avalanches in SAR imagery is time-consuming and prone to inter-annotator variability. This repository provides a semi-automatic tool that adapts the Segment Anything Model (SAM) to the remote sensing domain. By leveraging a custom prompt-generation pipeline and DEM-derived hillshades, our tool achieves a 60.28% speed-up in the annotation process compared to fully manual methods.Key Features:SAR + DEM Fusion: Integration of backscatter data with topographic context.Semi-Automatic Pipeline: Bounding-box-to-mask generation optimized for avalanche geometries.Operational Integration: Designed for seamless embedding into forecasting workflows.
⚙️ Installation
Clone the repository:Bashgit clone https://github.com/RiccardoGelato/AdaptingSAMToSARAvalancheDetection.git
cd AdaptingSAMToSARAvalancheDetection
Create the environment:Bashconda env create -f environment.yml
conda activate avalanche-sam
Download SAM Checkpoints:Download the sam_vit_h weights from the official SAM repository and place them in the ./weights/ folder.
🚀 Usage
1. Data PreparationPlace your SAR .tif files and corresponding DEM hillshades in the data/ directory.
2. Running the ToolTo start the semi-automatic segmentation interface:Bashpython main.py --input ./data/sample_avalanche.tif --hillshade ./data/sample_hillshade.tif
3. Training/Fine-tuning (Optional)If you wish to reproduce our domain adaptation experiments:Bashpython train.py --config configs/base_adaptation.yaml
📊 Results
Our experiments demonstrate that the tool significantly reduces cognitive load for experts while maintaining high segmentation quality.Statistical Significance: $p < 10^{-5}$ (Paired one-tailed t-test).Efficiency: Median annotation time reduced from ~120s to ~15s per image.
🖋️ Citation
If you find this work useful for your research, please cite:Snippet di codice@article{gelato2025adapting,
  title={Adapting Segment Anything Model (SAM) for Avalanche Detection in SAR Imagery},
  author={Gelato, Riccardo and Sgaravatti, Carlo},
  journal={Working Paper / Thesis},
  year={2025}
}
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
