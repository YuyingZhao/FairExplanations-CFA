for seed in 1 2 3 4 5
do
	for lr_epochs in 1e-2,1000
	do
		lr=${lr_epochs%,*};
		epochs=${lr_epochs#*,};
		for weight_decay in 1e-3 1e-4 1e-5
		do
			for dropout in 0.1 0.3 0.5
			do
				for lambda_ in 0 0.001 0.01 0.1 1 10
				do 
					python train.py --epochs=$epochs --dataset='german' \
					--seed=$seed --weight_decay=$weight_decay \
					--dropout=$dropout --lr=$lr \
				 	--lambda_=$lambda_ 
				done
			done
		done
	done
done
