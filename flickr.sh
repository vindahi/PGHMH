# #!/bin/bash

# 定义要测试的超参数和它们的值
declare -A hyperparameters
hyperparameters=(
  ["hyper_global"]="0.0001 0.01 1 10 100"
  ["hyper_local"]="0.0001 0.01 1 10 100"
  ["hyper_cls_sum"]="0.0001 0.01 1 10 100"
)

# 基础命令
base_command="python test.py \
  --is-train \
  --dataset flickr25k \
  --query-num 2000 \
  --train-num 10000 \
  --lr 0.001 \
  --rank 0 \
  --valid-freq 1 \
  --epochs 100 \
  --result-name \"results/Result_PromptHash_Flickr\""

# 遍历每个超参数
for hyper in "${!hyperparameters[@]}"; do
  # 获取当前超参数的所有取值
  IFS=' ' read -ra values <<< "${hyperparameters[$hyper]}"
  
  # 遍历每个取值
  for value in "${values[@]}"; do
    echo "Starting training on Flickr25K with $hyper=$value..."
    
    # 执行命令，替换当前超参数的值
    $base_command --$hyper $value
    
    echo "Finished training on Flickr25K with $hyper=$value."
    echo "----------------------------------------"
  done
done

echo "All training runs completed."


# echo "Starting training on Flickr25K..."
# python test.py \
#   --is-train \
#   --dataset flickr25k \
#   --query-num 2000 \
#   --train-num 10000 \
#   --lr 0.001 \
#   --rank 0 \
#   --valid-freq 1 \
#   --epochs 100 \
#   --result-name "result/Result_PromptHash_Flickr" \
#   --hyper-recon 0.001 
# echo "Finished training on Flickr25K."


