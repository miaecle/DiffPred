from collect_predictions import *

data_dirs = [
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_477/ex0/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_20/ex0/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_202/ex0/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_142/ex1/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_273/ex2/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_480/ex0/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_100/ex3/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_100/ex4/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_854/ex0/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_854/ex1/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_839/ex1/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/different_wells/12well/line_839/ex0/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/different_wells/24well/line_975-839/ex0/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/different_differentiation/line_839/ex0/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_975/ex0/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/different_wells/12well/line_975/ex0/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/different_differentiation/line_975/ex0/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line1_3R/ex2_other_instrument/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line1_3R/ex2_prospective/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line1_3R/prospective_ex0/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/different_wells/12well/line1_3R/ex2-12well/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/different_wells/24well/line1_3R/ex0-24well/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/different_differentiation/line1_3R/ex0/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line1_3R/ex_Vizgen-slide7-neg/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line1_3R/ex_Vizgen-slide17-pos/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line1_3R/ex_Vizgen-slide15-pos/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_institutions/ex_UofT/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_institutions/ex_UTexas/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_institutions/ex_UColorado/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_drugs/ex4_Benzopyrene/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_drugs/ex_multi_line_Benzopyrene/0-to-0",
"/oak/stanford/groups/jamesz/zqwu/iPSC_data/VALIDATION/line_drugs/ex0_Valproate/0-to-0",
]

model_paths = {
    'ex-valid-pred': '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-inf_ex/bkp.model',
    'ex-test-pred': '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-inf_ex/test-bkp.model',
    'psp-valid-pred': '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-inf_ex_pspnet/bkp.model',
    'psp-test-pred': '/oak/stanford/groups/jamesz/zqwu/iPSC_data/model_save/ex_split/0-to-inf_ex_pspnet/test-bkp.model',
}

gen_fn = CustomGenerator
for model_name, model_path in model_paths.items():
    model = get_model(model_path)
    for data_dir in data_dirs:
        valid_gen = get_data_gen(data_dir, gen_fn, batch_size=8, with_label=False)
        for target_day in [15, 16, 17, 18]:
            input_filter = partial(filter_for_inf_predictor, day_min=1, day_max=12)
            input_transform = partial(augment_fixed_end, end=target_day)

            pred_save_dir = data_dir.replace('0-to-0', f'{model_name}-to-{target_day}')
            if not os.path.exists(pred_save_dir):
                print(pred_save_dir, flush=True)
                collect_preds(valid_gen, model, pred_save_dir, input_transform=input_transform, input_filter=input_filter)
