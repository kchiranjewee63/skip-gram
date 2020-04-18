# skip-gram

Implementation of skip-gram in PyTorch with following features:

* Negative Sampling
* Frequent Words Subsampling
* Batch Training
* Similarity Evaluation 
* Analogy Evaluation 
* Visualization  

Provide corpus path and other parameters in ``` config.py ``` and run ``` python train.py ``` for training.

To evaluate generated word embeddings run ``` python similarity_eval.py ``` for similarity evaluation and ``` python analogy_eval.py ``` for analogy evaluation.

To visualize the embeddings run ``` python plot.py ```.