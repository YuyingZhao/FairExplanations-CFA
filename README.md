# Fairness-and-Explainability-CFA
This repository is the implementation of our proposed Comprehensive Fairness Algorithm (CFA) and our two explanation fairness evaluation metrics proposed in "Fairness and Explainability: Bridging the Gap Towards Fair Model Explanations".

## Motivation

<img width="550" alt="Screen Shot 2022-11-29 at 7 40 26 PM" src="https://user-images.githubusercontent.com/58016786/204686597-9d8b3dbd-b00d-47ce-8f6d-fa717819357c.png">

Although various fairness metrics have been proposed, almost all of them are directly computed based on the outputs. Accordingly, the quantified unfairness could only reflect the result-oriented bias while ignoring the potential bias caused by the decision-making procedure. The neglect of such procedure-oriented bias would restrict the debiasing methods to provide a comprehensive solution that is fair in both prediction and procedure. Therefore, we aim to bridge the gap between fairness and explaninability towards fair model explanations.

## Framework
<img width="550" alt="Screen Shot 2022-11-29 at 7 40 46 PM" src="https://user-images.githubusercontent.com/58016786/204686631-b9635ab1-05c4-4a29-9f89-8f89a10a35e7.png">


We propose a Comprehensive Fairness Algorithm (CFA) with multiple objectives, aiming to achieve good performance on utility, traditional and explanation fairness. The framework is built upon the traditional training with extra distance loss to minimize the hidden representation distances from two groups based on their original features and masked features. The former helps improve traditional fairness while two components together improve explanation fairness.


## Configuration
The default version of python we use is 3.8.8. The versions of pytorch and geometric are as follows:
```linux
- Pytorch 1.10.1 with Cuda 11.3
- Pytorch-geometric 2.0.3
```
For other packages, please import the environment from CFA_environment.yml with command:
```linux
conda env create -f CFA_environment.yml
```

## File structure
- For ease of understanding, the directory structure and their functions are as follows:
```linux
├── dataset
│   ├── bail (called Recidivism in the paper)
│   ├── german
│   ├── math
│   ├── por
├── mlp.py
├── parse.py
├── train.py (train CFA and record the validation records)
├── test.py (test CFA on the test dataset with the obtained best hyperparameters)
├── utils.py
├── explanation_metrics.py (explanation fairness evaluation)
├── run_german.sh (command to train CFA on german dataset, similar for other datasets)
├── run_best.sh (command to test CFA)
├── scripts
│   ├── Best Validation.ipynb (obtain the best hyperparameter)
│   ├── Test Result.ipynb (obtain the test result)
```
## If you are interested in the explanation fairness evaluation
Two proposed explanation fairness metrics are in _explanation_metrics.py_ where we include the explainer (graphlime) and the interpreter for evaluation. Since our framework is model-agonostic, here the main input to the interpreter is the inputs and outputs from the model and also ground truth label for calculating utility performance and sensitive label for calculating fairness. Feel free to play around with other models once they have the same input and output formats.

## If you are interested in the CFA algorithm
1. Use _train.py_ to obtain validation logs (refer to _run_german.sh_ of how to run train.py)
2. Use _Best Valiation.ipynb_ to obtain the best validation hyperparameters
3. Use the obtained hyperparameters and _test.py_ to obtain results in test dataset (refer to _run_best.sh_ of how to run _test.py_)
4. Use _Test Result.ipynb_ to print the test results


## Acknowledgement: The code is developed based on part of the code in the following papers:
```linux
[1] Huang, Qiang, Makoto Yamada, Yuan Tian, Dinesh Singh, and Yi Chang. "Graphlime: Local interpretable model explanations for graph neural networks." IEEE Transactions on Knowledge and Data Engineering (2022).
[2] Agarwal, Chirag, Himabindu Lakkaraju, and Marinka Zitnik. "Towards a unified framework for fair and stable graph representation learning." In Uncertainty in Artificial Intelligence, pp. 2114-2124. PMLR, 2021.
```

