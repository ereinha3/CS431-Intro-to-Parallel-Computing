make
rm -rf thread*
for i in $(seq 1 20);
do
        sbatch prefix07.srun
	sbatch prefix14.srun
	sbatch prefix21.srun
	sbatch prefix28.srun
done
