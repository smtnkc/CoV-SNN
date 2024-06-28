#!/bin/bash

# Default values for the arguments
loss_name="TripletLoss"
neg_set="delta"
pooling_mode="max"
batch_size=32
epochs=10
lr=1e-3

# Arrays of values to iterate over
margins=(0.2 0.5 1.0 2.0 5.0)
dropouts=(0.0 0.1 0.2 0.5 0.8)
relus=(0.0 0.1 0.2 0.5 1.0)

# Loop over each combination of margin, dropout, and relu
for margin in "${margins[@]}"; do
    for dropout in "${dropouts[@]}"; do
        for relu in "${relus[@]}"; do
            # Construct the command to run the Python script with the arguments
            cmd="python train_test_triplet.py --loss_name $loss_name --neg_set $neg_set"
            cmd+=" --pooling_mode $pooling_mode --batch_size $batch_size --epochs $epochs"
            cmd+=" --lr $lr --relu $relu --dropout $dropout --margin $margin --gstats $1"

            # Run the command
            echo "************************************************************"
            echo "Running command: $cmd"
            echo "************************************************************"
            eval $cmd
        done
    done
done
