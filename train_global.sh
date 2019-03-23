for v in 1
do
python3 -m global_model.train \
        --batch_size=64 --experiment_name=corefmerge \
        --training_name=group_global/global_trans_400_1_4_model_v$v \
        --evaluation_minutes=40 --nepoch_no_imprv=6 \
        --span_emb="boundaries" \
        --dropout=0.5 \
        --entity_extension=extension_entities \
        --no_pre_training \
        --train_datasets=aida_train \
        --ed_datasets=aida_dev_z_aida_test --ed_val_datasets=0 --ed_test_datasets=1
done