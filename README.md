# Protein Sequence Structure Prediction

## requisites

python>=3.6.

tensorflow==1.8.

sklearn==0.19.1 

matplotlib==2.0.2

## train the model:

python MNNN/food_train.py

## inferrence/tesing:

(1) paste target data into MNNN/foodData/testData.txt. (format: "ABC... %name", as example)

(2) run MNNN/foodData/interGen.py to reformat the input data.

(3) run MNNN/inferrence.py to get model output.

(4) run MNNN/generateOutput.py to reformat output data based on configurations.

(5) run MNNN/results/npyRes/printPreds.py to display the output model predictions, output path: MNNN/results/npyRes/output.txt  