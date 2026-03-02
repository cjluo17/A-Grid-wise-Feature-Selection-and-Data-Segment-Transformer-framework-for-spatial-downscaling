# GRACE Terrestrial Water Storage Downscaling Framework

A Grid-wise Feature Selection and Data Segment Transformer(GFSDSformer) framework for spatial downscaling of terrestrial water storage change data.

## Authors
1. Chuanjiang Luo,
	School of Architecture and Civil Engineering, Chengdu University, Chengdu 610106, China
	Email: luochuanjiang@cdu.edu.cn

2. Lilu Cui,
	School of Architecture and Civil Engineering, Chengdu University, Chengdu 610106, China
	Email: cuililu@cdu.edu.cn

3. Bo Zhong,
	School of Geodesy and Geomatics, Wuhan University, Wuhan 430079, China
	Email: bzhong@sgg.whu.edu.cn
	
## Key Features
- Transformer-based temporal modeling with causal self-attention
- Grid-wise RF feature importance analysis
- Automatic optimal window search (DS strategy)
- Parallelized computation for large-scale grids

## Usage
```matlab
% Simple example
[HighResPred, LowResPred] = Modeling( ...
    LRF, HRF, LREWH, ...
    LowResCoords, HighResCoords, ...
    Region);

CorrectedHigh = ResidualCorrect( ...
    LowResPred, HighResPred, LREWH);
```
## System Requirements

### Hardware
| Component      | Minimum              | Recommended          |
|----------------|----------------------|----------------------|
| RAM            | 4 GB DDR4            | 16 GB DDR4           |
| GPU            | NVIDIA GTX 1060 (4GB)| NVIDIA RTX 3070 (8GB)|
| Storage        | 50 GB HDD            | 1024 GB NVMe SSD     |

### Software
- MATLAB R2021b or later
- Deep Learning Toolbox
- Statistics and Machine Learning Toolbox
- Parallel Computing Toolbox

## Dataset Requirements
| Variable                             | Units  | Description              |
|--------------------------------------|--------|--------------------------|
| GLDAS TWSC (from GLDAS)              | mm     | Simulated TWSC           |
| SM, ET, PRE, NDVI,...(from ERA5-land)| mm     | hydrological variables   |
| TWSC(from CSR, GFZ, and JPL)         | cm     | Target variable          |

## Execution Pipeline
1.  Data Preparation
2.  Compute RF Feature Importance: `compute_rf_importance.m`
3.  Grid-wise Feature Selection: `gfs_select_features.m`
4.  Data Segment strategy: `find_best_window.m`
5.  Sample Construction: `DS_Split.m`
6.  Transformer Training and Prediction: `Modeling.m`
7.  Residual Correction: `ResidualCorrect.m`