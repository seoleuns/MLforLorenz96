# MLforLorenz96

## Citation

The related manuscript has been published in *Physical Review E*:

> Seoleun Shin, "Diffusion model-based ensemble generation for highly nonlinear
> systems with limited ensemble size", Phys. Rev. E **114**, 024218 (2026).
> https://doi.org/10.1103/xq7z-9cg5

Please cite this article if you use the idea in this code.

### BibTeX

```bibtex
@article{Shin2026,
  author  = {Shin, Seoleun},
  title   = {Diffusion model-based ensemble generation for highly nonlinear systems with limited ensemble size},
  journal = {Physical Review E},
  volume  = {114},
  pages   = {024218},
  year    = {2026},
  doi     = {10.1103/xq7z-9cg5}
}
```

This repository will be updated further if it is necessary.

The directory "Model" provides codes for the default Diff96DDPM/DDIM prediction for Lorenz '96 system.
It also contains LAFDiff but this directory may move in other place for a better structure of file hierarchy.

The Framework is based on the DiffSTG at https://github.com/wenhaomin/DiffSTG

Changes for the applications to the Lorenz '96 system have been made:

1. Addition of an Attention Mechanism 
2. Unnormalized scale of data
3. Addition of data assimilation process
4. A New Sampling strategy during the reverse processes in DDIM and DDPM 
5. Use of diverse evaluation Metrics for Time series Data

The size of Numpy Array Datasets is quite large and I will provide them upon request or will upload necessary source code to generate them.

(Contact Info: Seoleun Shin, seoleuns@kriss.re.kr)

