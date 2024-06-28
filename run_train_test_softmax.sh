#!/bin/bash

# Default values for the arguments
loss_name="SoftmaxLoss"
neg_set="other"
pooling_mode="max"
concat="CD"
num_labels=2
batch_size=32
epochs=10
lr=1e-3

# Arrays of values to iterate over

conf_thresholds=(0.95 0.99 0.995) #(0.0 0.95 0.975 0.99 0.995)
relus=(0.0 0.1 0.2 0.5 1.0)
dropouts=(0.0 0.1 0.2 0.5 0.8)

# Loop over each combination of conf_threshold, dropout, and relu
for conf_threshold in "${conf_thresholds[@]}"; do
    for dropout in "${dropouts[@]}"; do
        for relu in "${relus[@]}"; do

            # Construct the command to run the Python script with the arguments
            cmd="python train_test_softmax.py --loss_name $loss_name --neg_set $neg_set --pooling_mode $pooling_mode"
            cmd+=" --concat $concat --num_labels $num_labels --batch_size $batch_size --epochs $epochs"
            cmd+=" --lr $lr --relu $relu --dropout $dropout --conf_threshold $conf_threshold --gstats $1"

            # Run the command
            echo "************************************************************"
            echo "Running command: $cmd"
            echo "************************************************************"
            eval $cmd
        done
    done
done
