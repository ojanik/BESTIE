# **BESTIE**  
A framework for creating end-to-end-optimized summary statistics for binned forward-folding likelihood fits.

---

## **Installation**

To use **BESTIE**, ensure the following dependencies are installed:

1. **JAX**:  
   Install JAX according to your specific GPU configuration. Follow the official installation guide: [JAX Installation Documentation](https://jax.readthedocs.io/en/latest/installation.html).  

2. **PyTorch (CPU version)**:  
   Install the CPU version of PyTorch. You can find installation instructions here: [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).

3. **NNMFit**:  
   Required for calculating flux weights. Alternatively, you can implement a new weighting method if needed.

---

## **Usage**

### **Optimizing Binning**  
Use the provided training script to optimize the binning:

```bash
python training.py --config_path BESTIE/configs/general_config_SPL.yaml \
                   --dataset_config_path BESTIE/configs/dataset_standard.yaml \
                   --output_dir ./ \
                   --sample \
                   --name SPL_test \
                   --trainstep_pbar
```

Modify the config files such that it fits your purpose. 

### **Plot results**  

To visualize results and export them to the dataframe, use the following script:

```bash
python plot_results.py --model_path ./SPL_test \
                       --make_weighted_hist \
                       --make_unweighted_hist \
                       --make_2D_scatter \
                       --save_df
```
