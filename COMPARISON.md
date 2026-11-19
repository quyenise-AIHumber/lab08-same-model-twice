# PyTorch vs. Keras: Framework Comparison

## Scenario and methodology

Lakeshore Bike Share needs to select a machine-learning framework for a weekly bike-demand forecasting process maintained by a two-person team. One member knows PyTorch, while the other knows Keras. Future integration with pretrained Hugging Face models is also possible.

I implemented the same 8-32-32-1 neural-network architecture in both frameworks, using identical prepared data, MSE loss, Adam optimizer, learning rate (0.001), batch size (64), and 80 training epochs. Both implementations were evaluated using seeds 42, 43, and 44.

## Observed differences

**1. Prediction accuracy.** Keras achieved a mean test MAE of 4.523 bikes, compared with 5.350 for PyTorch, representing approximately 15.5% lower prediction error. Keras had lower test MAE for all three seeds.

**2. Training speed.** PyTorch averaged 4.326 seconds per run, compared with 12.352 seconds for Keras. PyTorch was approximately 2.86 times faster in this environment.

**3. Consistency.** PyTorch had a test MAE standard deviation of 0.174, compared with 0.670 for Keras, indicating more consistent results across the three seeds.

**4. Development approach.** Keras handles training through `compile()` and `fit()`, while PyTorch requires an explicit training loop. Keras provides a simpler high-level workflow, whereas PyTorch offers more direct control over optimization and debugging.

## Recommendation

I recommend **Keras provisionally** for Lakeshore Bike Share's weekly forecasting workflow. Its lower average prediction error and simpler training interface are valuable for a small team. The observed additional training time is unlikely to be operationally significant for a weekly job.

The trade-offs are slower training, greater variation across seeds, and less direct control over custom training behavior. The comparison is also limited to three seeds on synthetic data; different initialization and batch-order behavior may contribute to the measured differences.

## Condition for reconsideration

I would reconsider PyTorch if the team adopts pretrained Hugging Face models that require or strongly favor PyTorch, or if broader evaluation on real rental data eliminates Keras's accuracy advantage.

The recommendation should be revisited using real operational data before production deployment.