# to train Naive CNN

python run.py --task 1

# to evaluate Naive CNN

python code/run.py --task 1 --load-checkpoint code/checkpoints/your_model1/041921-235651/your.weights.e049-acc0.0000.h5 --evaluate

# to train our model

python run.py --task 2

# to evaluate our model

python code/run.py --task 2 --load-checkpoint code/checkpoints/your_model2/042021-015938/your.weights.e049-acc0.0000.h5 --evaluate

# to train VGG

python run.py --task 3

# to evaluate VGG

python code/run.py --task 3 --load-checkpoint code/checkpoints/vgg_model/041821-014654/vgg.weights.e049-acc0.0000.h5 --evaluate