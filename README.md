# Table Understanding with Tree-based Attention (TUTA)

Code for understanding generally structured real-world tables.


## Models
We provide three variants of pre-trained TUTA models: TUTA (-implicit), TUTA-explicit, and TUTA-base.
These pre-trained TUTA variants can be downloaded from:
* [TUTA](https://drive.google.com/file/d/1pEdrCqHxNjGM4rjpvCxeAUchdJzCYr1g/view?usp=sharing)
* [TUTA-explicit](https://drive.google.com/file/d/1FPwn2lQKEf-cGlgFHr4_IkDk_6WThifW/view?usp=sharing)
* [TUTA-base](https://drive.google.com/file/d/1j5qzw3c2UwbVO7TTHKRQmTvRki8vDO0l/view?usp=sharing)


## Pre-training
To pre-train a TUTA model, simply run
```bash
python train.py                                           \
--dataset_paths="../dataset.pt"                              \
--pretrained_model_path="${tuta_model_dir}/tuta.bin"      \
--output_model_path="${tuta_model_dir}/trained-tuta.bin"

# to enable a quick test, one can run
python train.py  --batch_size 1  --chunk_size 10  --buffer_size 10  --report_steps 1  --total_steps 20

# to enable multi-gpu distributed training, additionally specify 
--world_size 4  --gpu_ranks 0 1 2 3
```
Do make sure that the number of input `dataset_paths` is no less that the `world_size` (i.e. number of `gpu_ranks`). \
One can find more adjustable arguments in the main procedure.


## Downstream tasks

### __Cell Type Classification (CTC)__
To perform the task of cell type classification at downstream: 
- for data processing, use `SheetReader` in the reader.py and `CtcTokenizer` in the tokenizer.py; 
- for fine-tuning, use the `CtcHead` and `TUTA(base)forCTC` in the ./model/ directory.

### __Table Type Classification (TTC)__
To perform the task of table type classification at downstream: 
- for data processing, use `SheetReader` in the reader.py and `TtcTokenizer` in the tokenizer.py; 
- for fine-tuning, use the `TtcHead` and `TUTA(base)forTTC` in the ./model/ directory.


## Data Pre-processing
For a sample raw table file input, run
```bash
# for SpreadSheet
python prepare.py                          \
--input_dir ../data/pretrain/spreadsheet   \
--source_type sheet                        \
--output_path ../dataset.pt

# for WikiTable
python prepare.py                                      \
--input_path ../data/pretrain/wiki-table-samples.json  \
--source_type wiki                                     \
--output_path ../dataset.pt

# for WDCTable
python prepare.py                         \
--input_dir ../data/pretrain/wdc          \
--source_type wdc                         \
--output_path ../dataset.pt
```

will generate a semi-processed version for pre-training inputs.

Input this data file as an argument into the pre-training script, then the data-loader will dynamically process for three pre-training objectives, namely Masked Language Model (MLM), Cell-Level Cloze(CLC), and Table Context Retrieval (TCR).
