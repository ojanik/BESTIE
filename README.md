# **BESTIE**  
A framework for creating end-to-end-optimized summary statistics for binned forward-folding likelihood fits.

---

## **Installation**

To use **BESTIE** with GPU support please install JAX according to your specific GPU configuration. Follow the official installation guide: [JAX Installation Documentation](https://jax.readthedocs.io/en/latest/installation.html).  

---

## **Usage**

### **Inputs**  
BESTIE requires a dataframe containing the following information for each individual event:
- Input variables for the neural network
- Event weights
- Gradients of events weights with respect to the signal and nuisance parameters
  

Modify the config file such that it fits your purpose. 


