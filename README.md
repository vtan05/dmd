# Motion to Dance Music Generation
[**"Motion to Dance Music Generation using Latent Diffusion Model"**](https://dmdproject.github.io/) - Official PyTorch Implementation  

![teaser](https://github.com/DMDproject/DMDproject.github.io/blob/main/static/images/main_figure_dmd.jpg)


## Installation

This code was tested on `Ubuntu 20.04.2 LTS` and requires:

* Python 3.8
* CUDA capable GPU
* Download [data and models](https://drive.google.com/file/d/1FRZY-RWiSno_yo7MYYzSri5DWEWSGukG/view?usp=sharing)

```bash
pip install -r requirements.txt
```

## Preprocess data
### Generate mel spectrograms
```bash
python audio_to_images.py
```

### Generate concatenated motion and genre features
```bash
python norm_motion.py
```

## Training and inference
### Train latent diffusion model using pre-trained VAE
```bash
python train_unet_latent.py
```

### Generate samples then normalize loudness
```bash
python eval_cdcd.py --gen_audio=True
python post_process.py
```

## Evaluation
```bash
python eval_cdcd.py # beat coverage score, beat hit score, and FAD
python bas_cdcd.py  # beat align score
python genre.py     # genre KLD (get pretrained model from https://github.com/PeiChunChang/MS-SincResNet)
```

## Acknowledgments

We would like to thank [Joel Casimiro](https://sites.google.com/eee.upd.edu.ph/joelcasimiro) for helping in creating our preview image.  
We would also like to thank the following contributors that our code is based on: [Audio-Diffusion](https://github.com/teticio/audio-diffusion/), [EDGE](https://github.com/Stanford-TML/EDGE), [Bailando](https://github.com/lisiyao21/Bailando), [AIST++](https://github.com/google-research/mint), [MS-SincResNet](https://github.com/PeiChunChang/MS-SincResNet).
