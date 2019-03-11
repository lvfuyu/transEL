for v in 1 2 3
do
python3 -m model.train
        --batch_size=4 --experiment_name=global_datasets \
		--training_name=group_gloal/global_trans_400_1_4_model_v$v \
		--evaluation_minutes=100 --nepoch_no_imprv=6 \
		--span_emb="boundaries" \
		--fast_evaluation=True  \
		--dropout=0.5 \
		--train_datasets=aida_train \
		--ed_datasets=aida_dev_z_aida_test_z_aida_train --ed_val_datasets=0 \
done
