
<div align="center">
<h1> VAD </h1>
<h3>Vectorized Scene Representation for Efficient Autonomous Driving</h3>

[Bo Jiang](https://github.com/rb93dett)<sup>1</sup>\*, [Shaoyu Chen](https://scholar.google.com/citations?user=PIeNN2gAAAAJ&hl=en&oi=sra)<sup>1</sup>\*, Qing Xu<sup>2</sup>, [Bencheng Liao](https://github.com/LegendBC)<sup>1</sup>, Jiajie Chen<sup>2</sup>, [Helong Zhou](https://scholar.google.com/citations?user=wkhOMMwAAAAJ&hl=en&oi=ao)<sup>2</sup>, [Qian Zhang](https://scholar.google.com/citations?user=pCY-bikAAAAJ&hl=zh-CN)<sup>2</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>, [Chang Huang](https://scholar.google.com/citations?user=IyyEKyIAAAAJ&hl=zh-CN)<sup>2</sup>, [Xinggang Wang](https://xinggangw.info/)<sup>1 :email:</sup>
 
<sup>1</sup> Huazhong University of Science and Technology, <sup>2</sup> Horizon Robotics

(\*) equal contribution, (<sup>:email:</sup>) corresponding author.

</div>


<div align="center">
<img src="./framework.png" />
</div>

## Introduction
TL;DR: VAD is a vectorized paradigm for end-to-end autonomous driving.

Autonomous driving requires a comprehensive understanding of the surrounding environment for reliable trajectory planning. Previous works rely on dense rasterized scene representation (e.g., agent occupancy and semantic map) to perform planning, which is computationally intensive and misses the instance-level structure information. In this paper, we propose VAD, an end-to-end  vectorized paradigm for autonomous driving, which models the driving scene as fully vectorized representation. The proposed vectorized paradigm has two significant advantages. On one hand, VAD exploits the vectorized agent motion and map elements as explicit instance-level planning constraints which effectively improves planning safety. On the other hand, VAD runs much faster than previous end-to-end planning methods by getting rid of computation-intensive rasterized representation and hand-designed post-processing steps. VAD achieves state-of-the-art end-to-end planning performance on the nuScenes dataset, outperforming the previous best method by a large margin (reducing the average collision rate by 48.4\%).  Besides, VAD greatly improves the inference speed (up to 9.3x), which is critical for the real-world deployment of an autonomous driving system. Code and models will be released for facilitating future research.

<div align="center">
<img src="./vis.png" />
</div>

