# dgm-online-select
Official repository of the paper "An Online Learning Approach to Prompt-based Selection of Generative Models and LLMs" (ICML 2025)

[Xiaoyan Hu](https://yannxiaoyanhu.github.io), [Ho-fung Leung](http://www.cse.cuhk.edu.hk/~lhf/), [Farzan Farnia](https://www.cse.cuhk.edu.hk/~farnia/Home.html) [[Paper](https://arxiv.org/pdf/2410.13287)]

![Figure](https://github.com/yannxiaoyanhu/dgm-online-select/blob/main/Examples.png)
![Figure](https://github.com/yannxiaoyanhu/dgm-online-select/blob/main/Prompt_based_select.png)

## Usage Examples

PAK-UCB-poly3: ```python test.py --learner pak-ucb --kernel_method poly --kernel_para_gamma 10.0```

RFF-UCB-RBF: ```python test.py --learner rff-ucb --kernel_method rbf --kernel_para_gamma 5.0 --num_rff_dim 200```

## Acknowledgements

The authors would like to acknowledge the following repositories:

1. OpenCLIP: [https://github.com/mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)
2. MS-COCO dataset: [https://cocodataset.org/#home](https://cocodataset.org/#home)
3. Stable Diffusion: [https://huggingface.co/CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4), [https://huggingface.co/stabilityai/stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2)
4. UniDiffuser v1: [https://huggingface.co/thu-ml/unidiffuser-v1](https://huggingface.co/thu-ml/unidiffuser-v1)
5. PixArt-$\alpha$: [https://github.com/PixArt-alpha/PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha)
6. DeepFloyd: [https://huggingface.co/DeepFloyd](https://huggingface.co/DeepFloyd)


## Citation
```
@misc{hu2025onlinelearningapproachpromptbased,
      title={An Online Learning Approach to Prompt-based Selection of Generative Models}, 
      author={Xiaoyan Hu and Ho-fung Leung and Farzan Farnia},
      year={2025},
      eprint={2410.13287},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.13287}, 
}
```
