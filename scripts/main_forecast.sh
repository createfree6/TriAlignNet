
all_models=("TriAlignNet")

GPU=0

root_path=./data

seeds=(1111)

datasets=("Environment" "Climate" "Economy" "Agriculture" "Climate" "Energy" "Health" "Security" "Traffic" "weather_hs_4hours" "weather_ny_4hours" "weather_sf_4hours" )

current_dir=$(pwd)


text_emb=8  # 12 4 8
pred_lengths=(6 12 18 24)

for seed in "${seeds[@]}"
do
    for model_name in "${all_models[@]}"
    do
        for dataset in "${datasets[@]}"
        do
            data_path=${dataset}.csv
            model_id=$(basename ${root_path})

            for pred_len in "${pred_lengths[@]}"
            do
                echo "Running model $model_name with root $root_path, data $data_path, and pred_len $pred_len"
                CUDA_VISIBLE_DEVICES=${GPU} python -u run.py \
                    --task_name long_term_forecast \
                    --is_training 1 \
                    --root_path $root_path \
                    --data_path $data_path \
                    --model_id ${model_id}_${seed}_24_${pred_len}_fullLLM_${use_fullmodel}_${dataset} \
                    --model $model_name \
                    --data custom \
                    --seq_len 24 \
                    --label_len 12 \
                    --pred_len $pred_len \
                    --text_emb $text_emb \
                    --des Exp \
                    --batch_size 32 \
                    --learning_rate 0.0001 \
                    --d_model 512 \
                    --e_layers 1 \
                    --seed $seed \
                    --save_name result_environment_itransformer_gpt2 \
                    --llm_model GPT2 \
                    --huggingface_token 'XXXXXX' \
                    --train_epochs 100 \
                    --patience 10
            done
        done
    done
done

