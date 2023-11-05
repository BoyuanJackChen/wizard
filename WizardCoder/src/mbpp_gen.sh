# Activate your Conda environment
conda activate wizard

export TRANSFORMERS_CACHE="/data/bc3194/huggingface_cache"
model="WizardLM/WizardCoder-Python-7B-V1.0"
temp=0.0
max_len=2048
pred_num=1
num_seqs_per_iter=1
decoding_style="greedy"  # Or "sample"
greedy_decode="True"
mbpp_path="../data/mbpp.test.jsonl"
output_path=preds/MBPP_T${temp}_N${pred_num}

mkdir -p ${output_path}
echo 'Output path: '$output_path
echo 'Model to eval: '$model

# 164 problems, 21 per GPU if GPU=8
index=0
gpu_num=4
num_questions=500
num_question_per_gpu=125
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * $num_question_per_gpu))
  end_index=$(((i + 1) * $num_question_per_gpu))

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$gpu python mbpp_gen.py --model ${model} --start_index ${start_index} --end_index ${end_index} --temperature ${temp} \
      --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} --max_len ${max_len} \
      --output_path ${output_path} --decoding_style ${decoding_style} --mbpp_path ${mbpp_path}
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done
