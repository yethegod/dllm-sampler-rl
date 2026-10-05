#!/bin/bash
# Serial ghx4-interactive driver for slurm/eval_pos_threshold_deltaai.sbatch (QoS: one
# interactive job per user, 2h max). Each srun takes 4 GH200s and runs two cells side by
# side on GPU pairs (0,1) and (2,3); a cell whose generations JSON already exists is
# skipped, so the chain can be restarted, and cells finished by a batch array element
# (or on Delta, same RESULTS on /work) are not redone.
#
#   nohup bash slurm/eval_pos_threshold_int_chain.sh [first_task] [last_task] \
#     > slurm/logs/posthr_int_chain.log 2>&1 &

cd /u/zsun9/dllm-sampler-rl
FIRST=${1:-0}
LAST=${2:-55}
RESULTS=${RESULTS:-/work/hdd/bhta/zsun9/eval_results/pos_threshold}
SCRIPT=slurm/eval_pos_threshold_deltaai.sbatch

# Same enumeration as the sbatch (sourced from it, so the two cannot drift).
source <(awk '/^CELLS=\(\)/,/^TASK=/' "$SCRIPT" | grep -v '^TASK=')

cell_done() {  # cell_done <task>
  local DS S T B NAME
  read -r DS S T B <<< "${CELLS[$1]}"
  NAME=baseline-fastdllm-t$T
  [ "$S" != "-" ] && NAME=$NAME-s$S
  compgen -G "$RESULTS/$DS/bl$B/$NAME/checkpoint-${NAME}_seed_42*/${DS}_*generations.json" > /dev/null
}

todo=()
for ((i = FIRST; i <= LAST; i++)); do
  if cell_done "$i"; then echo "task $i (${CELLS[$i]}) done already, skip"; else todo+=("$i"); fi
done
echo "=== $(date) ${#todo[@]} cells to run: ${todo[*]}"

for ((k = 0; k < ${#todo[@]}; k += 2)); do
  A=${todo[$k]}
  B=${todo[$((k + 1))]:-}
  echo "=== $(date) start tasks $A ${B:-(none)}"
  srun --account=bhta-dtai-gh --partition=ghx4-interactive --nodes=1 --ntasks-per-node=1 \
    --gpus-per-node=4 --cpus-per-task=32 --mem=200g --time=02:00:00 --job-name=posthr-int \
    bash -c "
      CUDA_VISIBLE_DEVICES=0,1 TASK=$A EVAL_MAIN_PROCESS_PORT=$((29600 + A * 2)) \
        bash $SCRIPT > slurm/logs/posthr-int_task$A.out 2>&1 &
      PA=\$!
      if [ -n '$B' ]; then
        CUDA_VISIBLE_DEVICES=2,3 TASK=$B EVAL_MAIN_PROCESS_PORT=$((29601 + ${B:-0} * 2)) \
          bash $SCRIPT > slurm/logs/posthr-int_task$B.out 2>&1 &
        PB=\$!
        wait \$PB; echo \"task $B rc=\$?\"
      fi
      wait \$PA; echo \"task $A rc=\$?\"
    "
  for t in $A $B; do
    if cell_done "$t"; then echo "=== $(date) task $t OK"; else echo "=== $(date) task $t FAILED (no generations json)"; fi
  done
done
echo "=== $(date) chain done"
