for v in 1
do
python3 -m global_model.train \
        --batch_size=16 --experiment_name=corefmerge \
        --training_name=group_global/global_trans_model_v$v \
        --evaluation_minutes=5 --nepoch_no_imprv=100 \
        --span_emb="boundaries" \
        --dropout=0.5 \
        --entity_extension=extension_entities \
        --no_pre_training \
        --no_use_local \
        --train_datasets=aida_train_gt \
        --ed_datasets=aida_dev_z_aida_test_z_ace2004_z_aquaint_z_clueweb_z_msnbc_z_wikipedia --ed_val_datasets=0 --ed_test_datasets=0_1_2_3_4_5_6
done