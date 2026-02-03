# Optimization-guided Data Curation for Imbalanced Classification using Optimal Decision Trees

This repo contains python scripts implementing the pipeline to train Optimal Classification Trees with improved splits and prediction for minority outcome class. 

**Abstract**
Rare-event detection in high-stakes domains such as identifying patients at risk of exceptionally high future medical spending, or flagging fraudulent transactions demands interpretable models, making decision trees a natural choice.
However, rare events imply extreme class imbalance, and decision trees are known to suffer under such conditions: the learned partitions collapse into shallow, majority-dominated structures that fail to capture minority-class signal, degrading both predictive performance and the model's value as an auditable decision tool.
We address this problem 
by formulating training data curation as a principled optimization problem that explicitly balances two objectives: retaining majority samples near the decision boundary (boundary support) and maintaining geometric coverage across the majority distribution.
Unlike existing sampling heuristics, this formulation provides a clear optimization target.
To scale to large datasets, we derive a two-stage approximation: a farthest-first $k$-center construction that inherits classical approximation guarantees, followed by exact bipartite matching on the resulting candidate pool.
Furthermore, we introduce a leaf-level evaluation metric that quantifies the residual discriminative power of one model within the subgroups induced by another, enabling direct comparison of tree partition quality beyond global accuracy.
Experiments on a large medical spending prediction task of kidney-disease patients and a credit card fraud benchmark demonstrate that decision trees trained on curated subsets achieve improved minority-class performance on held-out test sets and yield more informative partitions under the proposed leaf-level evaluation, compared to training on the original imbalanced data or standard data resampling baselines.

README_REQUIREMENTS.md contains instructions on how to use this framework.
