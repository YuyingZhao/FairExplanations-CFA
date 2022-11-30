for dataset in 'german' 'math' 'por' 'bail'
do
	for seed in 1 2 3 4 5
	do
		python test.py --dataset=$dataset --seed=$seed
	done
done