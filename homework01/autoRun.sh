cd critical
make
rm output/*
for i in $(seq 1 20);
do
        sbatch pi07.srun
	sbatch pi14.srun
	sbatch pi21.srun
	sbatch pi28.srun
done
cd ../atomic
make
rm output/*
for i in $(seq 1 20);
do
        sbatch pi07.srun
	sbatch pi14.srun
	sbatch pi21.srun
	sbatch pi28.srun
done
