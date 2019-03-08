for v in 1 2 3
do
python3 -m model.train  \
			                     --batch_size=4   --experiment_name=corefmerge \
								 --training_name=group_local/local_trans_400_1_2_model_v$v \
								 --ent_vecs_regularization=l2dropout  --evaluation_minutes=1 --nepoch_no_imprv=6 \
								 --span_emb="boundaries"  \
								 --dim_char=50 --hidden_size_char=50 --hidden_size_lstm=150 \
								 --nn_components=pem_trans_attention \
								 --fast_evaluation=True  \
								 --attention_ent_vecs_no_regularization=True --final_score_ffnn=0_0 \
								 --attention_R=10 --attention_K=100 \
								 --train_datasets=aida_train \
								 --ed_datasets=aida_dev_z_aida_test_z_aida_train --ed_val_datasets=0 \
								 --global_thr=0.001 --global_score_ffnn=0_0
done
