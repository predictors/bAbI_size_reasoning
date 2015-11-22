# bAbI size reasoning predictor

You can find a live demo at https://predictors.ai/bAbI_size_reasoning_predictor

This predictor answers questions contained in the bAbI size reasoning task.

It is based on the sample code available for the Keras library at https://github.com/fchollet/keras/blob/master/examples/babi_rnn.py

More info about the bAbI project at https://research.facebook.com/researchers/1543934539189348

The following words are allowed (case sensitive): "is", "Does", "in", "yes", "container", "fit", "no", "chocolates", ".", "chest", "?", "chocolate", "fits", "The", "than", "bigger", "box", "of", "Is", "suitcase", "the", "inside".

Input story example:

The box of chocolates fits inside the chest. The container is bigger than the chest. The box of chocolates is bigger than the chocolate. The chest fits inside the container. The suitcase is bigger than the chocolate. 


Input questions examples:


Does the chest fit in the chocolate?

Is the box of chocolates bigger than the container?

Does the chocolate fit in the chest?

Is the chocolate bigger than the chest?

Does the box of chocolates fit in the container?
