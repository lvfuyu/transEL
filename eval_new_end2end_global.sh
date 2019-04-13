python -m model.evaluate \
    --training_name=group_global/global_trans_new_end2end_model_v1 \
    --experiment_name=corefmerge \
    --entity_extension=extension_entities \
    --ed_datasets=aida_dev_z_aida_test_z_aida_train_z_ace2004_z_aquaint_z_msnbc_z_clueweb_z_wikipedia \
    --ed_val_datasets=0 --el_datasets="" --el_val_datasets=0
#