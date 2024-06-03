# Selective Copying Task with Mamba Model

This repository contains a simple implementation for reproducing the selective copying task with the Mamba model.



## Files

- `config.py`: Contains the configuration for training, dataset, and the Mamba model.
- `data_generator.py`: Contains the `torch_copying_data` function for generating a dataset for a selective copying task and the `generate_dataset` function for generating a dataset based on the provided configuration.
- `selective_copying_mamba.py`: Contains the main script for training and validating the Mamba model.



## Usage

1. Configure your training, dataset, and model parameters in `config.py`.
2. Run `selective_copying_mamba.py` to train and validate the model.



## Running the Scripts

To run the main script, use the following command:

```
python selective_copying_mamba.py
```



## Results

After training, you can view the results of the selective copying task in the terminal. Sample results might look like this:

```
2024-06-03 16:03:06,983 - Step [399995/400000], Loss: 0.0000, Accuracy: 100.00%
2024-06-03 16:03:06,988 - Step [399996/400000], Loss: 0.0000, Accuracy: 100.00%
2024-06-03 16:03:06,993 - Step [399997/400000], Loss: 0.0000, Accuracy: 100.00%
2024-06-03 16:03:06,999 - Step [399998/400000], Loss: 0.0000, Accuracy: 100.00%
2024-06-03 16:03:07,005 - Step [399999/400000], Loss: 0.0000, Accuracy: 100.00%
2024-06-03 16:03:07,010 - Step [400000/400000], Loss: 0.0000, Accuracy: 100.00%
2024-06-03 16:03:07,010 - Training completed in: 34.91 minutes
2024-06-03 16:03:07,013 - Validation Accuracy: 100.00%
```

The above results are obtained with sequences of length 100 for demonstration purposes. Similar results can be achieved with sequences of length 4096, but more training time will be required.



## Acknowledgments

We would like to thank the authors Dao and Gu for their work, as referenced in [this paper](https://arxiv.org/pdf/2312.00752.pdf), and for the model used in their [implementation](https://github.com/state-spaces/mamba).