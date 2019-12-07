Tried Various models by changing number of layers, number of filters, dropout value, Optimizer and with Batch Normalization
* here are TensorBoard data of loss functions of some models (we didn use Batch Normalisation in this models) 

Loss of various Functions in Tensorboard:
![](https://github.com/wizardboy2010/MNIST/blob/master/Lossofall.png)

![](https://github.com/wizardboy2010/MNIST/blob/master/losszoom.png)

Accuracy of those models varies drastically:
![](https://github.com/wizardboy2010/MNIST/blob/master/accall.png)

Out of all these variations mod1 that we constructed in [Submission.ipynb](https://github.com/wizardboy2010/MNIST/blob/master/Submission.ipynb) gave highest accuracy

For that model I tried variations like - Batch Normalisation, Batch Normalization + less Dropout

Loss of 3 models in Tensorboard:
![](https://github.com/wizardboy2010/MNIST/blob/master/finalloss.png)

Accuracy is:
![](https://github.com/wizardboy2010/MNIST/blob/master/finalacc.png)

Out of all these models mod1 with batch normalization in Dense layer gave a highest accuracy of 99.66% in Test.
